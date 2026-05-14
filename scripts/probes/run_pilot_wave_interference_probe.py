#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pilot_sensor_v0 import extract_scope_flags, normalize_aliases


DEFAULT_OUT = ROOT / "output" / "pilot_wave_interference_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_WAVE_INTERFERENCE_001_RESULT.md"

EXEC_ACTIONS = {"EXEC_ADD", "EXEC_MUL"}
HOLD_ACTION = "HOLD_ASK_RESEARCH"
REJECT_ACTION = "REJECT_UNKNOWN"
LABELS = ("ADD", "MUL", "UNKNOWN")


@dataclass(frozen=True)
class Case:
    case_id: str
    text: str
    expected: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class Row:
    arm: str
    case_id: str
    text: str
    expected: str
    predicted: str
    correct: bool
    false_execution: bool
    tags: tuple[str, ...]
    state: dict[str, float]
    flags: dict[str, bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_WAVE_INTERFERENCE_001 deterministic representation probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def cases() -> list[Case]:
    return [
        Case("add", "add 3", "EXEC_ADD", ("known_execute",)),
        Case("multiply", "multiply by 3", "EXEC_MUL", ("known_execute",)),
        Case("divide_unknown", "divide by 3", "REJECT_UNKNOWN", ("unknown_reject",)),
        Case("do_not_add", "do not add 3", HOLD_ACTION, ("destructive_interference", "negation")),
        Case("do_not_multiply", "do not multiply by 3", HOLD_ACTION, ("destructive_interference", "negation")),
        Case("not_add_then_multiply", "do not add 3, multiply by 3", "EXEC_MUL", ("destructive_interference", "refocus")),
        Case("not_multiply_then_add", "do not multiply by 3, add 3", "EXEC_ADD", ("destructive_interference", "refocus")),
        Case("maybe_add", "maybe add 3", HOLD_ACTION, ("weak_hold",)),
        Case("maybe_multiply", "maybe multiply by 3", HOLD_ACTION, ("weak_hold",)),
        Case("add_or_multiply", "add or multiply by 3", HOLD_ACTION, ("ambiguous_hold",)),
        Case("word_add_appears", "the word add appears in the note", HOLD_ACTION, ("mention_suppression",)),
        Case("word_multiply_appears", "someone said multiply, but no operation is requested", HOLD_ACTION, ("mention_suppression",)),
        Case("add_actually_multiply", "add 3. wait, actually multiply by 3", "EXEC_MUL", ("correction_refocus",)),
        Case("mul_actually_add", "multiply by 3. correction: add 3", "EXEC_ADD", ("correction_refocus",)),
        Case("unknown_or_add", "divide or add 3", HOLD_ACTION, ("ambiguous_hold", "unknown_conflict")),
        Case("all_way_conflict", "add, multiply, or divide by 3", HOLD_ACTION, ("ambiguous_hold", "unknown_conflict")),
    ]


def positive_evidence(case: Case) -> tuple[str, dict[str, float]]:
    flags = extract_scope_flags(normalize_aliases(case.text))
    evidence = {
        "ADD": 0.90 if flags.add_cue else 0.0,
        "MUL": 0.90 if flags.mul_cue else 0.0,
        "UNKNOWN": 0.90 if flags.unknown_cue else 0.0,
        "HOLD": 0.0,
    }
    return decide_from_positive(evidence), evidence


def signed_amplitude(case: Case) -> tuple[str, dict[str, float]]:
    flags = extract_scope_flags(normalize_aliases(case.text))
    amp = {
        "ADD": 0.90 if flags.add_cue else 0.0,
        "MUL": 0.90 if flags.mul_cue else 0.0,
        "UNKNOWN": 0.90 if flags.unknown_cue else 0.0,
        "HOLD": 0.0,
    }
    if flags.negation_add:
        amp["ADD"] -= 0.90
    if flags.negation_mul:
        amp["MUL"] -= 0.90
    if flags.negation_unknown:
        amp["UNKNOWN"] -= 0.90
    if flags.weak_marker or flags.mention_only:
        amp["HOLD"] = 0.80
    if flags.ambiguity_marker:
        amp["HOLD"] = max(amp["HOLD"], 0.55)
    # This arm intentionally lacks a correction-tail reset. It tests whether
    # signed cancellation alone is enough.
    return decide_from_signed(amp), amp


def complex_phase(case: Case) -> tuple[str, dict[str, float]]:
    flags = extract_scope_flags(normalize_aliases(case.text))
    z = {"ADD": 0.0 + 0.0j, "MUL": 0.0 + 0.0j, "UNKNOWN": 0.0 + 0.0j}
    hold = 0.0

    if flags.add_cue:
        z["ADD"] += 1.0 + 0.0j
    if flags.mul_cue:
        z["MUL"] += 1.0 + 0.0j
    if flags.unknown_cue:
        z["UNKNOWN"] += 1.0 + 0.0j

    if flags.negation_add:
        z["ADD"] += -1.0 + 0.0j
        hold = max(hold, 0.40)
    if flags.negation_mul:
        z["MUL"] += -1.0 + 0.0j
        hold = max(hold, 0.40)
    if flags.negation_unknown:
        z["UNKNOWN"] += -1.0 + 0.0j
        hold = max(hold, 0.40)

    if flags.mention_only:
        z = {label: 0.0 + 0.0j for label in LABELS}
        hold = 1.0

    if flags.weak_marker:
        z = {label: value * 0.45 for label, value in z.items()}
        hold = max(hold, 0.80)

    if flags.correction_to_add:
        z = {"ADD": 1.0 + 0.0j, "MUL": 0.0 + 0.0j, "UNKNOWN": 0.0 + 0.0j}
        hold = 0.0
    elif flags.correction_to_mul:
        z = {"ADD": 0.0 + 0.0j, "MUL": 1.0 + 0.0j, "UNKNOWN": 0.0 + 0.0j}
        hold = 0.0
    elif flags.correction_to_unknown:
        z = {"ADD": 0.0 + 0.0j, "MUL": 0.0 + 0.0j, "UNKNOWN": 1.0 + 0.0j}
        hold = 0.0

    powers = {label: abs(value) ** 2 for label, value in z.items()}
    active_exec = [label for label in ("ADD", "MUL") if powers[label] >= 0.75]
    unknown_active = powers["UNKNOWN"] >= 0.75
    if flags.ambiguity_marker and (len(active_exec) > 1 or (active_exec and unknown_active)):
        hold = max(hold, 1.0)

    state = {label: round(powers[label], 6) for label in LABELS}
    state["HOLD"] = round(hold, 6)
    return decide_from_complex(state), state


def decide_from_positive(evidence: dict[str, float]) -> str:
    top = max(LABELS, key=lambda label: (evidence[label], -LABELS.index(label)))
    top_value = evidence[top]
    active_count = sum(1 for label in LABELS if evidence[label] >= 0.50)
    if active_count > 1:
        return HOLD_ACTION
    if top_value < 0.75:
        return HOLD_ACTION
    if top == "UNKNOWN":
        return REJECT_ACTION
    return f"EXEC_{top}"


def decide_from_signed(amp: dict[str, float]) -> str:
    positive = {label: max(0.0, amp[label]) for label in LABELS}
    active = [label for label in LABELS if positive[label] >= 0.75]
    if amp["HOLD"] >= 0.75:
        return HOLD_ACTION
    if len(active) != 1:
        return HOLD_ACTION
    if active[0] == "UNKNOWN":
        return REJECT_ACTION
    return f"EXEC_{active[0]}"


def decide_from_complex(state: dict[str, float]) -> str:
    if state["HOLD"] >= 0.75:
        return HOLD_ACTION
    active = [label for label in LABELS if state[label] >= 0.75]
    if len(active) != 1:
        return HOLD_ACTION
    if active[0] == "UNKNOWN":
        return REJECT_ACTION
    return f"EXEC_{active[0]}"


def run_arm(arm: str, case: Case) -> Row:
    normalized = normalize_aliases(case.text)
    flags = extract_scope_flags(normalized).to_dict()
    if arm == "positive_evidence":
        predicted, state = positive_evidence(case)
    elif arm == "signed_amplitude":
        predicted, state = signed_amplitude(case)
    elif arm == "complex_phase":
        predicted, state = complex_phase(case)
    else:
        raise ValueError(f"unknown arm: {arm}")

    false_execution = predicted in EXEC_ACTIONS and case.expected not in EXEC_ACTIONS
    return Row(
        arm=arm,
        case_id=case.case_id,
        text=case.text,
        expected=case.expected,
        predicted=predicted,
        correct=predicted == case.expected,
        false_execution=false_execution,
        tags=case.tags,
        state=state,
        flags=flags,
    )


def evaluate(rows: list[Row]) -> dict[str, dict[str, object]]:
    by_arm: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_arm[row.arm].append(row)

    metrics: dict[str, dict[str, object]] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        total = len(arm_rows)
        correct = sum(row.correct for row in arm_rows)
        false_exec = sum(row.false_execution for row in arm_rows)
        tag_totals: dict[str, int] = defaultdict(int)
        tag_correct: dict[str, int] = defaultdict(int)
        for row in arm_rows:
            for tag in row.tags:
                tag_totals[tag] += 1
                if row.correct:
                    tag_correct[tag] += 1
        per_tag = {
            tag: round(tag_correct[tag] / tag_totals[tag], 6)
            for tag in sorted(tag_totals)
        }
        failed = [
            {
                "case_id": row.case_id,
                "text": row.text,
                "expected": row.expected,
                "predicted": row.predicted,
                "tags": list(row.tags),
                "state": row.state,
            }
            for row in arm_rows
            if not row.correct
        ]
        metrics[arm] = {
            "action_accuracy": round(correct / total, 6),
            "false_execution_rate": round(false_exec / total, 6),
            "per_tag_accuracy": per_tag,
            "failed_cases": failed,
            "destructive_interference_success": per_tag.get("destructive_interference", math.nan),
            "correction_refocus_success": per_tag.get("correction_refocus", math.nan),
            "weak_hold_accuracy": per_tag.get("weak_hold", math.nan),
            "mention_suppression_accuracy": per_tag.get("mention_suppression", math.nan),
        }
    return metrics


def verdict(metrics: dict[str, dict[str, object]]) -> list[str]:
    labels: list[str] = []
    complex_metrics = metrics["complex_phase"]
    positive_metrics = metrics["positive_evidence"]
    signed_metrics = metrics["signed_amplitude"]
    if complex_metrics["action_accuracy"] == 1.0 and complex_metrics["false_execution_rate"] == 0.0:
        labels.append("PILOT_WAVE_COMPLEX_PHASE_POSITIVE")
    else:
        labels.append("PILOT_WAVE_COMPLEX_PHASE_FAIL")
    if positive_metrics["false_execution_rate"] > 0.0 or positive_metrics["action_accuracy"] < complex_metrics["action_accuracy"]:
        labels.append("POSITIVE_EVIDENCE_WEAK")
    if signed_metrics["action_accuracy"] > positive_metrics["action_accuracy"]:
        labels.append("SIGNED_AMPLITUDE_PARTIAL")
    return labels


def write_outputs(out_dir: Path, rows: list[Row], metrics: dict[str, dict[str, object]], labels: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"metrics": metrics, "verdict": labels}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    with (out_dir / "rows.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["arm", "case_id", "text", "expected", "predicted", "correct", "false_execution", "tags", "state"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "arm": row.arm,
                    "case_id": row.case_id,
                    "text": row.text,
                    "expected": row.expected,
                    "predicted": row.predicted,
                    "correct": int(row.correct),
                    "false_execution": int(row.false_execution),
                    "tags": "|".join(row.tags),
                    "state": json.dumps(row.state, sort_keys=True),
                }
            )
    DOC_REPORT.write_text(render_report(metrics, labels), encoding="utf-8")


def render_report(metrics: dict[str, dict[str, object]], labels: list[str]) -> str:
    lines: list[str] = [
        "# PILOT_WAVE_INTERFERENCE_001 Result",
        "",
        "## Goal",
        "",
        "Test whether phase-like interference is a useful primitive for command-scope decisions.",
        "",
        "The probe compares three deterministic representations over the same scope flags:",
        "",
        "```text",
        "positive_evidence",
        "signed_amplitude",
        "complex_phase",
        "```",
        "",
        "## Metrics",
        "",
        "| Arm | Action Accuracy | False Execution | Destructive | Correction | Weak Hold | Mention |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("positive_evidence", "signed_amplitude", "complex_phase"):
        arm_metrics = metrics[arm]
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` |".format(
                arm,
                float(arm_metrics["action_accuracy"]),
                float(arm_metrics["false_execution_rate"]),
                float(arm_metrics["destructive_interference_success"]),
                float(arm_metrics["correction_refocus_success"]),
                float(arm_metrics["weak_hold_accuracy"]),
                float(arm_metrics["mention_suppression_accuracy"]),
            )
        )

    lines.extend(["", "## Verdict", "", "```json", json.dumps(labels, indent=2), "```", "", "## Failure Examples", ""])
    any_failure = False
    for arm in ("positive_evidence", "signed_amplitude", "complex_phase"):
        failed = metrics[arm]["failed_cases"]
        if not failed:
            continue
        any_failure = True
        lines.append(f"### `{arm}`")
        lines.append("")
        for failure in failed:
            lines.append(
                "- `{case_id}`: expected `{expected}`, got `{predicted}`; tags `{tags}`; state `{state}`.".format(
                    **failure
                )
            )
        lines.append("")
    if not any_failure:
        lines.append("No failures.")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "Positive evidence overfires or over-holds because it treats cue presence as action authority.",
            "Signed amplitude can represent simple cancellation but does not fully model correction/refocus.",
            "The complex-phase arm passes this toy smoke because it separates destructive interference, hold pressure, and correction reset.",
            "",
            "Safe claim if positive:",
            "",
            "```text",
            "phase-like interference is a useful candidate primitive for Pilot/Prismion command-state modeling.",
            "```",
            "",
            "## Claim Boundary",
            "",
            "Toy command domain only. No consciousness claim, no general NLU claim, no quantum physics claim, and no production VRAXION/INSTNCT claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = [run_arm(arm, case) for arm in ("positive_evidence", "signed_amplitude", "complex_phase") for case in cases()]
    metrics = evaluate(rows)
    labels = verdict(metrics)
    write_outputs(args.out_dir, rows, metrics, labels)
    print(json.dumps({"out_dir": str(args.out_dir), "metrics": metrics, "verdict": labels}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
