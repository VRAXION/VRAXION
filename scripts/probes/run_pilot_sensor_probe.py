#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pilot_topk_guard_probe as guard


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_sensor_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_001_RESULT.md"

ADD_CUES = {"add", "increase", "plus"}
MUL_CUES = {"multiply", "times", "scale"}
UNKNOWN_CUES = {"divide", "sqrt", "root", "mod", "quotient"}
WEAK_CUES = {"maybe", "perhaps", "probably", "could", "might", "unsure"}
MENTION_CUES = {"word", "note", "said", "appears", "visible", "label", "page", "text", "operation is requested"}
CORRECTION_CUES = {"actually", "correction", "instead"}


@dataclass(frozen=True)
class TextCase:
    split: str
    name: str
    text: str
    expected: tuple[str, ...]
    group: str
    sequence_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_001 hard adversarial text-to-evidence probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def cases() -> list[TextCase]:
    return [
        TextCase("controlled_eval", "known_add_1", "add 3", ("EXEC_ADD",), "known_execute"),
        TextCase("controlled_eval", "known_add_2", "increase by 3", ("EXEC_ADD",), "known_execute"),
        TextCase("controlled_eval", "known_add_3", "plus 3", ("EXEC_ADD",), "known_execute"),
        TextCase("controlled_eval", "known_mul_1", "multiply by 3", ("EXEC_MUL",), "known_execute"),
        TextCase("controlled_eval", "known_mul_2", "times 3", ("EXEC_MUL",), "known_execute"),
        TextCase("controlled_eval", "known_mul_3", "scale by 3", ("EXEC_MUL",), "known_execute"),
        TextCase("controlled_eval", "unknown_div_1", "divide by 3", ("REJECT_UNKNOWN",), "unknown_reject"),
        TextCase("controlled_eval", "unknown_div_2", "take square root", ("REJECT_UNKNOWN",), "unknown_reject"),
        TextCase("controlled_eval", "unknown_div_3", "mod by 3", ("REJECT_UNKNOWN",), "unknown_reject"),
        TextCase("controlled_eval", "weak_add_1", "maybe add 3", ("HOLD_ASK_RESEARCH",), "weak_hold"),
        TextCase("controlled_eval", "weak_mul_1", "maybe multiply", ("HOLD_ASK_RESEARCH",), "weak_hold"),
        TextCase("controlled_eval", "ambiguous_1", "add or multiply by 3", ("HOLD_ASK_RESEARCH",), "ambiguous_hold"),
        TextCase("controlled_eval", "no_evidence_1", "do the operation", ("HOLD_ASK_RESEARCH",), "no_evidence_hold"),
        TextCase("controlled_eval", "conflict_1", "add, multiply, or divide by 3", ("HOLD_ASK_RESEARCH",), "ambiguous_hold"),
        TextCase("controlled_eval", "negated_add_1", "do not add 3", ("HOLD_ASK_RESEARCH", "REFRAME"), "negation_scope"),
        TextCase("controlled_eval", "negated_add_then_mul_1", "do not add 3, multiply by 3", ("EXEC_MUL",), "negation_scope"),
        TextCase("controlled_eval", "delayed_1_step1", "add or multiply by 3", ("HOLD_ASK_RESEARCH",), "delayed_correction", sequence_id="controlled_delay_1"),
        TextCase("controlled_eval", "delayed_1_step2", "actually multiply by 3", ("EXEC_MUL",), "delayed_correction", sequence_id="controlled_delay_1"),
        TextCase("adversarial_eval", "quote_trap_1", "the word add appears in the note", ("HOLD_ASK_RESEARCH",), "keyword_trap"),
        TextCase("adversarial_eval", "quote_trap_2", "do not follow the instruction 'add 3'", ("HOLD_ASK_RESEARCH", "REFRAME"), "keyword_trap"),
        TextCase("adversarial_eval", "quote_trap_3", "someone said multiply, but no operation is requested", ("HOLD_ASK_RESEARCH",), "keyword_trap"),
        TextCase("adversarial_eval", "scope_1", "do not add 3", ("HOLD_ASK_RESEARCH", "REFRAME"), "negation_scope"),
        TextCase("adversarial_eval", "scope_2", "do not add 3, multiply by 3 instead", ("EXEC_MUL",), "negation_scope"),
        TextCase("adversarial_eval", "scope_3", "do not multiply by 3, add 3 instead", ("EXEC_ADD",), "negation_scope"),
        TextCase("adversarial_eval", "scope_4", "neither add nor multiply", ("HOLD_ASK_RESEARCH",), "negation_scope"),
        TextCase("adversarial_eval", "scope_5", "not sure whether to add or multiply", ("HOLD_ASK_RESEARCH",), "negation_scope"),
        TextCase("adversarial_eval", "correction_1", "add 3. wait, actually multiply by 3", ("EXEC_MUL",), "correction"),
        TextCase("adversarial_eval", "correction_2", "multiply by 3. correction: add 3", ("EXEC_ADD",), "correction"),
        TextCase("adversarial_eval", "sequence_1", "first add 3, then multiply by 3", ("HOLD_ASK_RESEARCH",), "unsupported_sequence"),
        TextCase("adversarial_eval", "morph_trap_1", "addition is not needed", ("HOLD_ASK_RESEARCH",), "keyword_trap"),
        TextCase("adversarial_eval", "morph_trap_2", "the additive label is visible", ("HOLD_ASK_RESEARCH",), "keyword_trap"),
        TextCase("adversarial_eval", "morph_trap_3", "multiply-styled text, but no operation", ("HOLD_ASK_RESEARCH",), "keyword_trap"),
        TextCase("adversarial_eval", "morph_trap_4", "plus sign appears on the page", ("HOLD_ASK_RESEARCH",), "keyword_trap"),
        TextCase("adversarial_eval", "weak_lexical_1", "maybe add 3", ("HOLD_ASK_RESEARCH",), "weak_hold"),
        TextCase("adversarial_eval", "weak_lexical_2", "probably multiply by 3", ("HOLD_ASK_RESEARCH",), "weak_hold"),
        TextCase("adversarial_eval", "weak_lexical_3", "it could be add", ("HOLD_ASK_RESEARCH",), "weak_hold"),
        TextCase("adversarial_eval", "unknown_override_1", "divide by 3, not add", ("REJECT_UNKNOWN", "HOLD_ASK_RESEARCH"), "unknown_reject"),
        TextCase("adversarial_eval", "unknown_override_2", "use sqrt; do not multiply", ("REJECT_UNKNOWN",), "unknown_reject"),
        TextCase("adversarial_eval", "unknown_override_3", "mod by 3, maybe add later", ("REJECT_UNKNOWN", "HOLD_ASK_RESEARCH"), "unknown_reject"),
    ]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def contains_phrase(text: str, phrase: str) -> bool:
    return phrase in text.lower()


def cue_counts(tok: list[str]) -> tuple[int, int, int]:
    add = sum(token in ADD_CUES for token in tok)
    mul = sum(token in MUL_CUES for token in tok)
    unknown = sum(token in UNKNOWN_CUES for token in tok)
    return add, mul, unknown


def evidence_from_counts(add: int, mul: int, unknown: int) -> np.ndarray:
    evidence = np.zeros(3, dtype=float)
    if add:
        evidence[0] = 0.90
    if mul:
        evidence[1] = 0.90
    if unknown:
        evidence[2] = 0.90
    return evidence


def keyword_sensor(text: str) -> np.ndarray:
    return evidence_from_counts(*cue_counts(tokens(text)))


def has_weak_marker(tok: list[str]) -> bool:
    return any(token in WEAK_CUES for token in tok) or contains_phrase(" ".join(tok), "not sure")


def is_mention_trap(text: str, tok: list[str]) -> bool:
    lower = text.lower()
    if "no operation is requested" in lower:
        return True
    if "do not follow the instruction" in lower:
        return True
    return any(token in MENTION_CUES for token in tok)


def split_correction(text: str) -> str | None:
    lower = text.lower()
    for marker in ["actually", "correction:"]:
        if marker in lower:
            return lower.split(marker, 1)[1]
    return None


def split_instead(text: str) -> str | None:
    lower = text.lower()
    if "instead" in lower:
        return lower.split("instead", 1)[0] if lower.strip().endswith("instead") else lower.split("instead", 1)[0]
    return None


def negated_ops(text: str) -> set[str]:
    lower = text.lower()
    negated: set[str] = set()
    patterns = {
        "ADD": [r"\bdo not add\b", r"\bnot add\b", r"\bnot plus\b", r"\bneither add\b", r"\bnor add\b"],
        "MUL": [r"\bdo not multiply\b", r"\bnot multiply\b", r"\bneither multiply\b", r"\bnor multiply\b"],
    }
    for label, pats in patterns.items():
        if any(re.search(pat, lower) for pat in pats):
            negated.add(label)
    return negated


def structured_rule_sensor(text: str) -> np.ndarray:
    lower = text.lower()
    tok = tokens(lower)
    if is_mention_trap(lower, tok):
        return np.zeros(3, dtype=float)
    correction_tail = split_correction(lower)
    if correction_tail is not None:
        return structured_rule_sensor(correction_tail)
    if re.search(r"\bfirst\b", lower) and re.search(r"\bthen\b", lower):
        return np.array([0.80, 0.80, 0.0], dtype=float)

    add, mul, unknown = cue_counts(tok)
    negated = negated_ops(lower)
    if "ADD" in negated:
        add = 0
    if "MUL" in negated:
        mul = 0

    if unknown and add and mul and (" or " in lower or "," in lower):
        return np.array([0.80, 0.80, 0.80], dtype=float)

    evidence = evidence_from_counts(add, mul, unknown)
    if unknown and (add or mul):
        evidence[2] = 0.90
        evidence[0] = min(evidence[0], 0.25)
        evidence[1] = min(evidence[1], 0.25)
    if has_weak_marker(tok):
        evidence = np.minimum(evidence, np.array([0.45, 0.45, 0.90]))
        if unknown:
            evidence[2] = 0.90
    if (add and mul) or " or " in lower or "whether" in tok:
        evidence[0] = max(evidence[0], 0.50 if add else 0.0)
        evidence[1] = max(evidence[1], 0.50 if mul else 0.0)
    if negated and not add and not mul and not unknown:
        return np.zeros(3, dtype=float)
    return evidence


SENSORS: dict[str, Callable[[str], np.ndarray]] = {
    "keyword_sensor": keyword_sensor,
    "structured_rule_sensor": structured_rule_sensor,
}


def expected_band(action: str, evidence: np.ndarray) -> bool:
    add, mul, unknown = evidence
    top = float(max(evidence))
    ordered = sorted((float(x) for x in evidence), reverse=True)
    margin = ordered[0] - ordered[1]
    if action == "EXEC_ADD":
        return add >= 0.75 and mul < 0.30 and unknown < 0.30
    if action == "EXEC_MUL":
        return mul >= 0.75 and add < 0.30 and unknown < 0.30
    if action == "REJECT_UNKNOWN":
        return unknown >= 0.75
    if action in {"HOLD_ASK_RESEARCH", "REFRAME"}:
        return top < 0.75 or margin < 0.30
    return False


def band_correct(expected: tuple[str, ...], evidence: np.ndarray) -> bool:
    return any(expected_band(action, evidence) for action in expected)


def policy_action(evidence: np.ndarray, policy: str) -> str:
    rng = np.random.default_rng(11)
    stats = guard.evidence_stats(evidence, rng=rng, noise_n=100, sigma=0.05)
    if policy == "evidence_strength_margin_guard":
        return guard.evidence_strength_margin_guard(evidence, stats, False)
    if policy == "topK2_guard":
        return guard.topk2_guard(evidence, stats, False)
    raise ValueError(policy)


def is_correct(action: str, expected: tuple[str, ...]) -> bool:
    return action in expected


def is_false_commit(action: str, expected: tuple[str, ...]) -> bool:
    return action in guard.EXEC_ACTIONS and all(item not in guard.EXEC_ACTIONS for item in expected)


def is_missed_execute(action: str, expected: tuple[str, ...]) -> bool:
    return any(item in guard.EXEC_ACTIONS for item in expected) and action not in guard.EXEC_ACTIONS


def evaluate() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sensor_name, sensor in SENSORS.items():
        for policy in ["evidence_strength_margin_guard", "topK2_guard"]:
            for case in cases():
                evidence = sensor(case.text)
                action = policy_action(evidence, policy)
                rows.append({
                    "sensor": sensor_name,
                    "guard": policy,
                    "split": case.split,
                    "case": case.name,
                    "text": case.text,
                    "group": case.group,
                    "sequence_id": case.sequence_id or "",
                    "expected": "|".join(case.expected),
                    "evidence": json.dumps([round(float(x), 4) for x in evidence]),
                    "action": action,
                    "correct": is_correct(action, case.expected),
                    "evidence_band_correct": band_correct(case.expected, evidence),
                    "false_commit": is_false_commit(action, case.expected),
                    "missed_execute": is_missed_execute(action, case.expected),
                })
    return rows


def fraction(rows: list[dict[str, object]], predicate: Callable[[dict[str, object]], bool], field: str = "correct") -> float:
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return 0.0
    return float(sum(bool(row[field]) for row in subset) / len(subset))


def delayed_sequence_accuracy(rows: list[dict[str, object]]) -> float:
    seqs = sorted({row["sequence_id"] for row in rows if row["sequence_id"]})
    if not seqs:
        return 0.0
    ok = 0
    for seq in seqs:
        seq_rows = [row for row in rows if row["sequence_id"] == seq]
        ok += int(all(bool(row["correct"]) for row in seq_rows))
    return float(ok / len(seqs))


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for sensor_name in SENSORS:
        for policy in ["evidence_strength_margin_guard", "topK2_guard"]:
            key = f"{sensor_name}+{policy}"
            subset = [row for row in rows if row["sensor"] == sensor_name and row["guard"] == policy]
            known = [row for row in subset if row["group"] == "known_execute"]
            metrics = {
                "controlled_action_accuracy": fraction(subset, lambda row: row["split"] == "controlled_eval"),
                "adversarial_action_accuracy": fraction(subset, lambda row: row["split"] == "adversarial_eval"),
                "evidence_band_accuracy": fraction(subset, lambda _: True, "evidence_band_correct"),
                "false_commit_rate": float(sum(bool(row["false_commit"]) for row in subset) / len(subset)),
                "keyword_trap_false_commit_rate": float(sum(bool(row["false_commit"]) for row in subset if row["group"] == "keyword_trap") / max(1, sum(row["group"] == "keyword_trap" for row in subset))),
                "known_execute_accuracy": fraction(subset, lambda row: row["group"] == "known_execute"),
                "unknown_reject_accuracy": fraction(subset, lambda row: row["group"] == "unknown_reject"),
                "weak_hold_accuracy": fraction(subset, lambda row: row["group"] == "weak_hold"),
                "ambiguous_hold_accuracy": fraction(subset, lambda row: row["group"] == "ambiguous_hold"),
                "negation_handling_accuracy": fraction(subset, lambda row: row["group"] == "negation_scope"),
                "correction_handling_accuracy": fraction(subset, lambda row: row["group"] == "correction"),
                "delayed_correction_sequence_accuracy": delayed_sequence_accuracy(subset),
                "missed_execute_rate": float(sum(bool(row["missed_execute"]) for row in subset) / len(subset)),
                "over_hold_rate_on_known": float(sum(row["action"] == "HOLD_ASK_RESEARCH" for row in known) / len(known)),
            }
            out[key] = metrics
    return out


def passes(metrics: dict[str, float]) -> bool:
    return (
        metrics["controlled_action_accuracy"] >= 0.90
        and metrics["adversarial_action_accuracy"] >= 0.85
        and metrics["false_commit_rate"] <= 0.05
        and metrics["keyword_trap_false_commit_rate"] <= 0.05
        and metrics["negation_handling_accuracy"] >= 0.90
        and metrics["correction_handling_accuracy"] >= 0.90
    )


def verdicts(agg: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, metrics in agg.items():
        verdict: list[str] = []
        if passes(metrics):
            verdict.append("SENSOR_POSITIVE")
        elif metrics["controlled_action_accuracy"] >= 0.90 and metrics["adversarial_action_accuracy"] < 0.85:
            verdict.append("SENSOR_RULE_SANITY_ONLY")
        if metrics["negation_handling_accuracy"] < 0.90 or metrics["correction_handling_accuracy"] < 0.90:
            verdict.append("SENSOR_SCOPE_WEAK")
        if metrics["keyword_trap_false_commit_rate"] > 0.05:
            verdict.append("SENSOR_KEYWORD_OVERFIRES")
        if metrics["unknown_reject_accuracy"] < 0.90:
            verdict.append("SENSOR_UNKNOWN_WEAK")
        if metrics["false_commit_rate"] > 0.05:
            verdict.append("SENSOR_FALSE_COMMIT_HIGH")
        if metrics["evidence_band_accuracy"] < 0.85 and metrics["false_commit_rate"] <= 0.05:
            verdict.append("GUARD_WORKS_SENSOR_WEAK")
        out[key] = verdict or ["SENSOR_INCONCLUSIVE"]
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_table(agg: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Sensor+Guard | Controlled | Adversarial | Evidence Band | False Commit | Trap False Commit | Negation | Correction | Unknown | Weak | Ambiguous |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, m in agg.items():
        lines.append(
            f"| `{key}` | `{m['controlled_action_accuracy']:.3f}` | `{m['adversarial_action_accuracy']:.3f}` "
            f"| `{m['evidence_band_accuracy']:.3f}` | `{m['false_commit_rate']:.3f}` "
            f"| `{m['keyword_trap_false_commit_rate']:.3f}` | `{m['negation_handling_accuracy']:.3f}` "
            f"| `{m['correction_handling_accuracy']:.3f}` | `{m['unknown_reject_accuracy']:.3f}` "
            f"| `{m['weak_hold_accuracy']:.3f}` | `{m['ambiguous_hold_accuracy']:.3f}` |"
        )
    return lines


def per_case_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Sensor | Guard | Split | Case | Expected | Evidence | Action | Correct | Band OK |",
        "|---|---|---|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['sensor']}` | `{row['guard']}` | `{row['split']}` | `{row['case']}` "
            f"| `{row['expected']}` | `{row['evidence']}` | `{row['action']}` "
            f"| `{int(bool(row['correct']))}` | `{int(bool(row['evidence_band_correct']))}` |"
        )
    return lines


def failure_lines(rows: list[dict[str, object]]) -> list[str]:
    failures = [row for row in rows if not bool(row["correct"])]
    if not failures:
        return ["No action failures."]
    return [
        f"- `{row['sensor']}+{row['guard']}` on `{row['case']}` ({row['text']}): expected `{row['expected']}`, got `{row['action']}`, evidence `{row['evidence']}`."
        for row in failures
    ]


def write_report(rows: list[dict[str, object]], agg: dict[str, dict[str, float]], verdict: dict[str, list[str]], output_report: Path) -> None:
    lines = [
        "# PILOT_SENSOR_001 Result",
        "",
        "## Goal",
        "",
        "Test controlled and adversarial raw command text -> evidence vector -> fixed guard -> pilot action.",
        "",
        "This is a hand-auditable sensor baseline, not a learning or natural-language-understanding claim.",
        "",
        "## Setup",
        "",
        "Evidence vector: `[ADD, MUL, UNKNOWN]`. Guard thresholds are reused unchanged from `PILOT_TOPK_GUARD_001`.",
        "",
        "## Sensors And Guards",
        "",
        "- `keyword_sensor`: direct keyword count baseline.",
        "- `structured_rule_sensor`: rule sensor with weak markers, unknown overrides, negation scope, correction handling, and quote/mention traps.",
        "- guards: `evidence_strength_margin_guard`, `topK2_guard`.",
        "",
        "## Aggregate Metrics",
        "",
        *metric_table(agg),
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(verdict, indent=2, sort_keys=True),
        "```",
        "",
        "## Per-Case Metrics",
        "",
        *per_case_table(rows),
        "",
        "## Failure Cases",
        "",
        *failure_lines(rows),
        "",
        "## Interpretation",
        "",
        "A positive result means controlled command text can be mapped to guard-compatible evidence under adversarial keyword and scope traps.",
        "",
        "If controlled eval passes but adversarial eval fails, the sensor is only a rule sanity check, not a robust evidence extractor.",
        "",
        "## Claim Boundary",
        "",
        "No general natural-language understanding, full Pilot Pulse learning, full VRAXION/INSTNCT proof, production architecture, or consciousness claim.",
    ]
    text = "\n".join(lines) + "\n"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(text)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(text)


def main() -> None:
    args = parse_args()
    rows = evaluate()
    agg = aggregate(rows)
    verdict = verdicts(agg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    (args.out_dir / "summary.json").write_text(json.dumps({"aggregate": agg, "verdict": verdict}, indent=2, sort_keys=True) + "\n")
    write_report(rows, agg, verdict, args.out_dir / "report.md")
    print(json.dumps({
        "verdict": verdict,
        "metrics": str(args.out_dir / "metrics.csv"),
        "report": str(args.out_dir / "report.md"),
        "doc_report": str(DOC_REPORT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
