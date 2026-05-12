#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pilot_sensor_probe as sensor_probe


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_sensor_lexicon_extension_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_LEXICON_EXTENSION_001_RESULT.md"


@dataclass(frozen=True)
class Case:
    split: str
    name: str
    text: str
    expected: tuple[str, ...]
    phenomenon_tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_LEXICON_EXTENSION_001 explicit alias lexicon probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def cases() -> list[Case]:
    return [
        Case("strict_alias", "increment", "increment by 9", ("EXEC_ADD",), "strict_synonym"),
        Case("strict_alias", "raise_value", "raise the value by 9", ("EXEC_ADD",), "strict_synonym"),
        Case("strict_alias", "product", "product with 9", ("EXEC_MUL",), "strict_synonym"),
        Case("strict_alias", "halve", "halve it", ("REJECT_UNKNOWN",), "strict_synonym"),
        Case("strict_alias", "exponentiate", "exponentiate by 3", ("REJECT_UNKNOWN",), "strict_synonym"),
        Case("scope_alias", "mention_increment", "the word increment appears in the note", ("HOLD_ASK_RESEARCH",), "mention_trap"),
        Case("scope_alias", "neg_increment", "do not increment by 3", ("HOLD_ASK_RESEARCH",), "negation"),
        Case("scope_alias", "neg_product_then_add", "do not product with 3, add 3 instead", ("EXEC_ADD",), "negation"),
        Case("scope_alias", "weak_increment", "maybe increment by 3", ("HOLD_ASK_RESEARCH",), "weak"),
        Case("scope_alias", "amb_increment_product", "increment or product with 3", ("HOLD_ASK_RESEARCH",), "ambiguous"),
        Case("scope_alias", "correction_product_increment", "product with 3. correction: increment by 3", ("EXEC_ADD",), "correction"),
        Case("regression", "known_add", "add 3", ("EXEC_ADD",), "known"),
        Case("regression", "known_mul", "multiply by 3", ("EXEC_MUL",), "known"),
        Case("regression", "mention_add", "the word add appears in the note", ("HOLD_ASK_RESEARCH",), "mention_trap"),
    ]


def normalize_aliases(text: str) -> str:
    lower = text.lower()
    replacements = [
        (r"\braise the value\b", "add"),
        (r"\bincrement\b", "add"),
        (r"\bproduct\b", "multiply"),
        (r"\bhalve\b", "divide"),
        (r"\bexponentiate\b", "divide"),
    ]
    for pattern, repl in replacements:
        lower = re.sub(pattern, repl, lower)
    return lower


def base_sensor(text: str) -> np.ndarray:
    return sensor_probe.structured_rule_sensor(text)


def alias_sensor(text: str) -> np.ndarray:
    return sensor_probe.structured_rule_sensor(normalize_aliases(text))


def evaluate(sensor_name: str, sensor_fn) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in cases():
        evidence = sensor_fn(row.text)
        action = sensor_probe.policy_action(evidence, "evidence_strength_margin_guard")
        rows.append({
            "sensor": sensor_name,
            "split": row.split,
            "case": row.name,
            "text": row.text,
            "normalized_text": normalize_aliases(row.text) if sensor_name == "alias_extended_sensor" else row.text,
            "phenomenon_tag": row.phenomenon_tag,
            "expected_action": "|".join(row.expected),
            "student_evidence": json.dumps([round(float(x), 4) for x in evidence]),
            "student_action": action,
            "action_correct": action in row.expected,
            "false_commit": sensor_probe.is_false_commit(action, row.expected),
            "missed_execute": sensor_probe.is_missed_execute(action, row.expected),
        })
    return rows


def fraction(rows: list[dict[str, object]], predicate, field: str = "action_correct") -> float:
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return 0.0
    return float(sum(bool(row[field]) for row in subset) / len(subset))


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for sensor in sorted({str(row["sensor"]) for row in rows}):
        subset = [row for row in rows if row["sensor"] == sensor]
        out[sensor] = {
            "action_accuracy": fraction(subset, lambda _: True),
            "strict_synonym_accuracy": fraction(subset, lambda row: row["phenomenon_tag"] == "strict_synonym"),
            "scope_alias_accuracy": fraction(subset, lambda row: row["split"] == "scope_alias"),
            "regression_accuracy": fraction(subset, lambda row: row["split"] == "regression"),
            "false_commit_rate": fraction(subset, lambda _: True, "false_commit"),
            "missed_execute_rate": fraction(subset, lambda _: True, "missed_execute"),
        }
    return out


def verdicts(agg: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for sensor, m in agg.items():
        labels: list[str] = []
        if sensor == "base_structured_sensor":
            labels.append("BASE_STRICT_SYNONYM_WEAK" if m["strict_synonym_accuracy"] < 0.75 else "BASE_STRICT_SYNONYM_UNEXPECTED_PASS")
        else:
            labels.append(
                "LEXICON_EXTENSION_POSITIVE"
                if m["strict_synonym_accuracy"] >= 0.95 and m["scope_alias_accuracy"] >= 0.95 and m["false_commit_rate"] == 0.0
                else "LEXICON_EXTENSION_WEAK"
            )
        out[sensor] = labels
    out["global"] = (
        ["STRICT_SYNONYM_IS_LEXICON_COVERAGE"]
        if "LEXICON_EXTENSION_POSITIVE" in out.get("alias_extended_sensor", [])
        else ["STRICT_SYNONYM_REMAINS_OPEN"]
    )
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, object]], agg: dict[str, dict[str, float]], verdict: dict[str, list[str]], output_report: Path) -> None:
    failures = [row for row in rows if not bool(row["action_correct"])]
    lines = [
        "# PILOT_SENSOR_LEXICON_EXTENSION_001 Result",
        "",
        "## Goal",
        "",
        "Test whether the strict synonym failures are lexical coverage failures rather than guard or execution-policy failures.",
        "",
        "## Aggregate Metrics",
        "",
        "| Sensor | Action | Strict Syn | Scope Alias | Regression | False Commit | Missed Execute |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sensor, m in agg.items():
        lines.append(
            f"| `{sensor}` | `{m['action_accuracy']:.3f}` | `{m['strict_synonym_accuracy']:.3f}` "
            f"| `{m['scope_alias_accuracy']:.3f}` | `{m['regression_accuracy']:.3f}` "
            f"| `{m['false_commit_rate']:.3f}` | `{m['missed_execute_rate']:.3f}` |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(verdict, indent=2, sort_keys=True),
        "```",
        "",
        "## Failure Examples",
        "",
    ]
    if failures:
        for row in failures:
            lines.append(
                f"- `{row['sensor']}` `{row['split']}/{row['phenomenon_tag']}`: {row['text']} -> "
                f"expected `{row['expected_action']}`, got `{row['student_action']}`."
            )
    else:
        lines.append("No failures.")
    lines += [
        "",
        "## Interpretation",
        "",
        "A positive alias sensor result means the previous strict synonym failures are solved by explicit lexicon coverage in this toy command grammar.",
        "This is not semantic generalization; it is a bounded lexical normalizer in front of the existing scope-aware sensor.",
        "",
        "## Claim Boundary",
        "",
        "No general NLU, pretrained semantics, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.",
    ]
    text = "\n".join(lines) + "\n"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(text)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(text)


def main() -> None:
    args = parse_args()
    rows = evaluate("base_structured_sensor", base_sensor) + evaluate("alias_extended_sensor", alias_sensor)
    agg = aggregate(rows)
    verdict = verdicts(agg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    (args.out_dir / "summary.json").write_text(json.dumps({"aggregate": agg, "verdict": verdict}, indent=2, sort_keys=True) + "\n")
    write_report(rows, agg, verdict, args.out_dir / "report.md")
    print(json.dumps({
        "doc_report": str(DOC_REPORT),
        "metrics": str(args.out_dir / "metrics.csv"),
        "report": str(args.out_dir / "report.md"),
        "verdict": verdict,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
