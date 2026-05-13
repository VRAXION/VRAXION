"""Deterministic PilotSensor v0 baseline for toy command routing.

This module is intentionally parser-assisted and stdlib-only. It is a stable
baseline for command-text -> evidence -> guard -> locked skill execution, not a
learned natural-language-understanding component.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass


ADD_CUES = {"add", "increase", "plus"}
MUL_CUES = {"multiply", "times", "scale"}
UNKNOWN_CUES = {"divide", "sqrt", "root", "mod", "quotient"}
WEAK_CUES = {"maybe", "perhaps", "probably", "could", "might", "unsure"}
MENTION_CUES = {
    "word",
    "note",
    "said",
    "appears",
    "visible",
    "label",
    "page",
    "text",
}

EVIDENCE_LABELS = ("ADD", "MUL", "UNKNOWN")
EXEC_ACTIONS = {"EXEC_ADD", "EXEC_MUL"}
HOLD_ACTION = "HOLD_ASK_RESEARCH"
REJECT_ACTION = "REJECT_UNKNOWN"
STRENGTH_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.30


Evidence = tuple[float, float, float]


@dataclass(frozen=True)
class ScopeFlags:
    add_cue: bool = False
    mul_cue: bool = False
    unknown_cue: bool = False
    weak_marker: bool = False
    ambiguity_marker: bool = False
    mention_only: bool = False
    multi_step_unsupported: bool = False
    negation_add: bool = False
    negation_mul: bool = False
    negation_unknown: bool = False
    correction_present: bool = False
    correction_to_add: bool = False
    correction_to_mul: bool = False
    correction_to_unknown: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class PilotSensorResult:
    text: str
    normalized_text: str
    flags: ScopeFlags
    evidence: Evidence
    action: str
    operand: int | None
    value: int | None
    result: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "flags": self.flags.to_dict(),
            "evidence": self.evidence,
            "action": self.action,
            "operand": self.operand,
            "value": self.value,
            "result": self.result,
        }


def normalize_aliases(text: str) -> str:
    """Normalize bounded command aliases into the toy command lexicon."""
    lower = text.lower()
    replacements = [
        (r"\braise the value\b", "add"),
        (r"\bincrement\b", "add"),
        (r"\bproduct\b", "multiply"),
        (r"\bhalve\b", "divide"),
        (r"\bexponentiate\b", "divide"),
    ]
    for pattern, replacement in replacements:
        lower = re.sub(pattern, replacement, lower)
    return lower


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _cue_counts(tok: list[str]) -> tuple[int, int, int]:
    add = sum(token in ADD_CUES for token in tok)
    mul = sum(token in MUL_CUES for token in tok)
    unknown = sum(token in UNKNOWN_CUES for token in tok)
    return add, mul, unknown


def _has_weak_marker(tok: list[str]) -> bool:
    return any(token in WEAK_CUES for token in tok) or "not sure" in " ".join(tok)


def _is_mention_trap(text: str, tok: list[str]) -> bool:
    lower = text.lower()
    if "no operation is requested" in lower:
        return True
    if "do not follow the instruction" in lower:
        return True
    return any(token in MENTION_CUES for token in tok)


def _correction_tail(text: str) -> str | None:
    lower = text.lower()
    for marker in ("actually", "correction:"):
        if marker in lower:
            return lower.split(marker, 1)[1]
    return None


def _negated_ops(text: str) -> set[str]:
    lower = text.lower()
    negated: set[str] = set()
    patterns = {
        "ADD": [r"\bdo not add\b", r"\bnot add\b", r"\bnot plus\b", r"\bneither add\b", r"\bnor add\b"],
        "MUL": [r"\bdo not multiply\b", r"\bnot multiply\b", r"\bneither multiply\b", r"\bnor multiply\b"],
        "UNKNOWN": [r"\bnot divide\b", r"\bnot sqrt\b", r"\bnot root\b", r"\bnot mod\b", r"\bnot quotient\b"],
    }
    for label, pats in patterns.items():
        if any(re.search(pat, lower) for pat in pats):
            negated.add(label)
    return negated


def extract_scope_flags(normalized_text: str) -> ScopeFlags:
    """Extract deterministic scope/event flags from already-normalized text."""
    lower = normalized_text.lower()
    tok = tokens(lower)
    add, mul, unknown = _cue_counts(tok)
    negated = _negated_ops(lower)
    tail = _correction_tail(lower)
    tail_add = tail_mul = tail_unknown = 0
    if tail is not None:
        tail_add, tail_mul, tail_unknown = _cue_counts(tokens(tail))
    return ScopeFlags(
        add_cue=bool(add),
        mul_cue=bool(mul),
        unknown_cue=bool(unknown),
        weak_marker=_has_weak_marker(tok),
        ambiguity_marker=(" or " in lower) or ("whether" in tok) or (bool(add) and bool(mul)),
        mention_only=_is_mention_trap(lower, tok),
        multi_step_unsupported=bool(re.search(r"\bfirst\b", lower) and re.search(r"\bthen\b", lower)),
        negation_add="ADD" in negated,
        negation_mul="MUL" in negated,
        negation_unknown="UNKNOWN" in negated,
        correction_present=tail is not None,
        correction_to_add=bool(tail_add),
        correction_to_mul=bool(tail_mul),
        correction_to_unknown=bool(tail_unknown),
    )


def flags_to_evidence(flags: ScopeFlags) -> Evidence:
    """Map scope/event flags to fixed ADD/MUL/UNKNOWN evidence."""
    if flags.mention_only:
        return (0.0, 0.0, 0.0)
    if flags.multi_step_unsupported:
        return (0.80, 0.80, 0.0)
    if flags.correction_to_add:
        return (0.90, 0.0, 0.0)
    if flags.correction_to_mul:
        return (0.0, 0.90, 0.0)
    if flags.correction_to_unknown:
        return (0.0, 0.0, 0.90)

    add = flags.add_cue and not flags.negation_add
    mul = flags.mul_cue and not flags.negation_mul
    unknown = flags.unknown_cue and not flags.negation_unknown
    if flags.ambiguity_marker and add and mul and unknown:
        return (0.80, 0.80, 0.80)

    evidence = [0.0, 0.0, 0.0]
    if add:
        evidence[0] = 0.90
    if mul:
        evidence[1] = 0.90
    if unknown:
        evidence[2] = 0.90

    if unknown and (add or mul):
        evidence[0] = min(evidence[0], 0.25)
        evidence[1] = min(evidence[1], 0.25)
        evidence[2] = 0.90

    if flags.weak_marker:
        evidence[0] = min(evidence[0], 0.45)
        evidence[1] = min(evidence[1], 0.45)
        if unknown:
            evidence[2] = 0.90
        else:
            evidence[2] = min(evidence[2], 0.90)

    if flags.ambiguity_marker and (add or mul):
        if add:
            evidence[0] = max(evidence[0], 0.50)
        if mul:
            evidence[1] = max(evidence[1], 0.50)

    return (float(evidence[0]), float(evidence[1]), float(evidence[2]))


def _top_two(evidence: Evidence) -> tuple[int, float, int, float]:
    order = sorted(range(len(evidence)), key=lambda idx: (-float(evidence[idx]), idx))
    top1 = int(order[0])
    top2 = int(order[1])
    return top1, float(evidence[top1]), top2, float(evidence[top2])


def guard_policy(evidence: Evidence) -> str:
    """Fixed strength+margin guard over ADD/MUL/UNKNOWN evidence."""
    top1, strength, _, top2_strength = _top_two(evidence)
    if strength < STRENGTH_THRESHOLD:
        return HOLD_ACTION
    if strength - top2_strength < MARGIN_THRESHOLD:
        return HOLD_ACTION
    if EVIDENCE_LABELS[top1] == "UNKNOWN":
        return REJECT_ACTION
    return f"EXEC_{EVIDENCE_LABELS[top1]}"


def extract_operand(text: str) -> int | None:
    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return None
    return int(numbers[-1])


def execute_locked_skill(action: str, value: int, operand: int | None) -> int | None:
    if operand is None:
        return None
    if action == "EXEC_ADD":
        return value + operand
    if action == "EXEC_MUL":
        return value * operand
    return None


def run_pilot_sensor_v0(text: str, value: int | None = None) -> PilotSensorResult:
    normalized = normalize_aliases(text)
    flags = extract_scope_flags(normalized)
    evidence = flags_to_evidence(flags)
    action = guard_policy(evidence)
    operand = extract_operand(normalized)
    result = execute_locked_skill(action, value, operand) if value is not None else None
    return PilotSensorResult(
        text=text,
        normalized_text=normalized,
        flags=flags,
        evidence=evidence,
        action=action,
        operand=operand,
        value=value,
        result=result,
    )
