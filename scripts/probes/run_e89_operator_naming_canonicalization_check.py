#!/usr/bin/env python3
"""Checker for E89 Operator naming canonicalization.

This is a documentation/schema gate, not a training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument(
        "--summary",
        default="target/pilot_wave/e89_operator_naming_canonicalization/checker_summary.json",
    )
    args = parser.parse_args()

    naming_path = REPO_ROOT / "docs/research/OPERATOR_NAMING_AND_LIBRARY_SCHEMA_LOCK.md"
    cards_path = REPO_ROOT / "docs/research/OPERATOR_LIBRARY_CARDS.md"
    legacy_cards_path = REPO_ROOT / "docs/research/POCKET_LIBRARY_CARDS.md"
    result_path = REPO_ROOT / "docs/research/E89_OPERATOR_NAMING_CANONICALIZATION_AND_LIBRARY_SCHEMA_ALIAS_RESULT.md"
    contract_path = REPO_ROOT / "docs/research/E89_OPERATOR_NAMING_CANONICALIZATION_AND_LIBRARY_SCHEMA_ALIAS_CONTRACT.md"

    naming = read(naming_path)
    cards = read(cards_path)
    legacy = read(legacy_cards_path)
    result = read(result_path)
    contract = read(contract_path)

    failures: list[str] = []

    for path, text in [
        (naming_path, naming),
        (cards_path, cards),
        (legacy_cards_path, legacy),
        (result_path, result),
        (contract_path, contract),
    ]:
        if not text:
            failures.append(f"missing or empty: {path.relative_to(REPO_ROOT)}")

    required_naming_terms = [
        "Operator",
        "OperatorToken",
        "Operator Library",
        "Operator Manager",
        "Registry",
        "Pocket = legacy alias for Operator",
        "No Operator may directly write stable Flow/Ground state",
        "alpha_syncer",
        "α-Syncer",
        "T-Stab",
        "LocalGolden is not automatically Core or TrueGolden",
    ]
    for term in required_naming_terms:
        if term not in naming:
            failures.append(f"naming lock missing term: {term}")

    required_card_terms = [
        "canonical generic term = Operator",
        "legacy alias = Pocket",
        "type         = governed Operator artifact",
        "family       = Scribe",
        "family       = α-Syncer",
        "family       = Guard",
        "Current Operator Library Summary",
    ]
    for term in required_card_terms:
        if term not in cards:
            failures.append(f"operator cards missing term: {term}")

    legacy_required = [
        "Compatibility Alias",
        "OPERATOR_LIBRARY_CARDS.md",
        "Pocket = legacy alias for Operator",
    ]
    for term in legacy_required:
        if term not in legacy:
            failures.append(f"legacy card pointer missing term: {term}")

    result_required = [
        "decision = e89_operator_naming_schema_lock_confirmed",
        "Pocket -> Operator",
        "LocalGolden != Core",
        "E90_OPERATOR_CURRICULUM_EXPANSION",
    ]
    for term in result_required:
        if term not in result:
            failures.append(f"result missing term: {term}")

    summary = {
        "decision": (
            "e89_operator_naming_schema_lock_confirmed"
            if not failures
            else "e89_operator_naming_schema_lock_failed"
        ),
        "failure_count": len(failures),
        "failures": failures,
        "checked_files": [
            str(naming_path.relative_to(REPO_ROOT)),
            str(cards_path.relative_to(REPO_ROOT)),
            str(legacy_cards_path.relative_to(REPO_ROOT)),
            str(result_path.relative_to(REPO_ROOT)),
            str(contract_path.relative_to(REPO_ROOT)),
        ],
    }

    if args.write_summary:
        summary_path = REPO_ROOT / args.summary
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
