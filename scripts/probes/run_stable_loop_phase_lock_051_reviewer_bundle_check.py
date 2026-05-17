#!/usr/bin/env python3
"""Static checker for the 051 reviewer reproduction bundle.

The checker reads committed repository files only. It does not run cargo, does
not run the 050 full reproduction command, and does not write target artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROBE = "STABLE_LOOP_PHASE_LOCK_051_PAPER_REPRODUCTION_BUNDLE"

QUICK_COMMAND = (
    "python scripts/probes/run_stable_loop_phase_lock_051_reviewer_bundle_check.py --check-only"
)
FULL_REPRO_COMMAND = (
    "python scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py --out "
    "target/pilot_wave/stable_loop_phase_lock_051_reviewer_bundle/full_repro "
    "--seeds 2026,2027,2028 --train-examples 8192 --heldout-examples 4096 "
    "--ood-examples 4096 --heartbeat-sec 20"
)

EXPECTED_HASHES_PATH = (
    ROOT / "docs/research/STABLE_LOOP_PHASE_LOCK_050_EXPECTED_HASHES.json"
)

REQUIRED_DOCS = {
    "contract": "docs/research/STABLE_LOOP_PHASE_LOCK_051_PAPER_REPRODUCTION_BUNDLE_CONTRACT.md",
    "result": "docs/research/STABLE_LOOP_PHASE_LOCK_051_PAPER_REPRODUCTION_BUNDLE_RESULT.md",
    "readme": "docs/research/STABLE_LOOP_PHASE_LOCK_051_REVIEWER_README.md",
    "checklist": "docs/research/STABLE_LOOP_PHASE_LOCK_051_ARTIFACT_CHECKLIST.md",
    "claim_boundary": "docs/research/STABLE_LOOP_PHASE_LOCK_051_CLAIM_BOUNDARY.md",
    "limitations": "docs/research/STABLE_LOOP_PHASE_LOCK_051_LIMITATIONS.md",
    "ablation": "docs/research/STABLE_LOOP_PHASE_LOCK_051_ABLATION_NARRATIVE.md",
    "tables": "docs/research/STABLE_LOOP_PHASE_LOCK_051_TABLES.md",
}

REQUIRED_SCRIPT = "scripts/probes/run_stable_loop_phase_lock_051_reviewer_bundle_check.py"

UNRESOLVED_MARKERS = [
    "todo",
    "tbd",
    "placeholder",
    "pending measured",
    "coming soon",
]

BOUNDARY_PHRASES = [
    "production default training",
    "public beta promotion",
    "production api readiness",
    "full vraxion",
    "language grounding",
    "consciousness",
    "biological/flywire equivalence",
    "physical quantum behavior",
]

REQUIRED_REFERENCES = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_049_ADVERSARIAL_FROZEN_EVAL_CORPUS.jsonl",
    "instnct-core/examples/phase_lane_adversarial_frozen_eval_scale.rs",
    "scripts/probes/run_stable_loop_phase_lock_050_repro_audit.py",
    "docs/research/STABLE_LOOP_PHASE_LOCK_050_EXPECTED_HASHES.json",
    "docs/research/STABLE_LOOP_PHASE_LOCK_050_REPRODUCIBILITY_PACKAGE_AND_PAPER_AUDIT_RESULT.md",
    "scripts/probes/run_stable_loop_phase_lock_051_reviewer_bundle_check.py",
    "target/pilot_wave/stable_loop_phase_lock_051_reviewer_bundle/full_repro",
]

REQUIRED_VERDICTS = [
    "PAPER_REPRODUCTION_BUNDLE_POSITIVE",
    "REVIEWER_README_WRITTEN",
    "ARTIFACT_CHECKLIST_WRITTEN",
    "CLAIM_BOUNDARY_DOCUMENTED",
    "LIMITATIONS_DOCUMENTED",
    "ABLATION_NARRATIVE_WRITTEN",
    "PAPER_TABLES_INCLUDED",
    "REPRO_COMMANDS_INCLUDED",
    "EXPECTED_HASHES_REFERENCED",
    "REVIEWER_BUNDLE_CHECK_PASSES",
    "PRODUCTION_API_NOT_READY",
]

SECTION_TOKENS = {
    "readme": ["## Quick Check", "## Full Reproduction", "## Evidence Sources"],
    "checklist": ["## Source Inputs", "## 050 Outputs", "## 051 Reviewer Bundle", "## Pass/Fail Criteria"],
    "claim_boundary": ["## Supports", "## Does Not Support"],
    "limitations": ["## Main Limitations", "## Practical Limits"],
    "ablation": ["## What 049/050 Showed", "## Why The Controls Matter"],
    "tables": ["## Main Table", "## Ablation And Failure-Control Table", "## Leakage-Audit Table"],
    "result": ["## Static Smoke", "## Referenced 050 Evidence", "## Verdicts"],
    "contract": ["## Summary", "## Required Reviewer Commands", "## Verdicts"],
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_expected_hashes() -> dict[str, Any]:
    return json.loads(read_text(EXPECTED_HASHES_PATH))


def missing_required_files() -> list[str]:
    paths = [*REQUIRED_DOCS.values(), REQUIRED_SCRIPT, str(EXPECTED_HASHES_PATH.relative_to(ROOT))]
    return [path for path in paths if not (ROOT / path).exists()]


def invalid_doc_reasons(path: Path, text: str) -> list[str]:
    reasons: list[str] = []
    stripped = text.strip()
    if len(stripped) < 300:
        reasons.append("too_short")
    lower = stripped.lower()
    for marker in UNRESOLVED_MARKERS:
        if marker in lower:
            reasons.append(f"unresolved_marker:{marker}")
    return reasons


def validate_docs(docs: dict[str, str]) -> tuple[list[str], list[str]]:
    invalid_docs: list[str] = []
    missing_sections: list[str] = []
    for key, rel_path in REQUIRED_DOCS.items():
        text = docs.get(key, "")
        reasons = invalid_doc_reasons(ROOT / rel_path, text)
        if reasons:
            invalid_docs.append(f"{rel_path}:{','.join(reasons)}")
        for token in SECTION_TOKENS[key]:
            if token not in text:
                missing_sections.append(f"{rel_path}:{token}")
        lower = text.lower()
        for phrase in BOUNDARY_PHRASES:
            if phrase not in lower:
                missing_sections.append(f"{rel_path}:claim_boundary:{phrase}")
    return invalid_docs, missing_sections


def validate_commands(readme: str) -> list[str]:
    missing: list[str] = []
    if QUICK_COMMAND not in readme:
        missing.append(QUICK_COMMAND)
    if FULL_REPRO_COMMAND not in readme:
        missing.append(FULL_REPRO_COMMAND)
    return missing


def validate_hashes(bundle_text: str) -> list[str]:
    expected = load_expected_hashes()
    required_hashes = [
        expected["corpus"]["sha256_normalized_lf"],
        expected["runner"]["sha256_normalized_lf"],
    ]
    return [hash_value for hash_value in required_hashes if hash_value not in bundle_text]


def validate_references(bundle_text: str) -> list[str]:
    return [reference for reference in REQUIRED_REFERENCES if reference not in bundle_text]


def validate_verdicts(bundle_text: str) -> list[str]:
    return [verdict for verdict in REQUIRED_VERDICTS if verdict not in bundle_text]


def production_flags_ok(bundle_text: str) -> bool:
    lower = bundle_text.lower()
    forbidden_true = [
        "production_default_training_enabled = true",
        '"production_default_training_enabled": true',
        "public_beta_promoted = true",
        '"public_beta_promoted": true',
        "production_api_ready = true",
        '"production_api_ready": true',
    ]
    if any(token in lower for token in forbidden_true):
        return False
    return all(phrase in lower for phrase in BOUNDARY_PHRASES)


def derive_verdicts(check_pass: bool, failures: dict[str, list[str]], prod_ok: bool) -> list[str]:
    if check_pass:
        return REQUIRED_VERDICTS[:]
    verdicts = ["REVIEWER_BUNDLE_CHECK_FAILS", "PRODUCTION_API_NOT_READY"]
    if "docs/research/STABLE_LOOP_PHASE_LOCK_051_REVIEWER_README.md" in failures["missing_files"]:
        verdicts.append("REVIEWER_README_MISSING")
    if "docs/research/STABLE_LOOP_PHASE_LOCK_051_CLAIM_BOUNDARY.md" in failures["missing_files"]:
        verdicts.append("CLAIM_BOUNDARY_MISSING")
    if "docs/research/STABLE_LOOP_PHASE_LOCK_051_LIMITATIONS.md" in failures["missing_files"]:
        verdicts.append("LIMITATIONS_MISSING")
    if failures["missing_commands"]:
        verdicts.append("REPRO_COMMAND_MISSING")
    if failures["missing_hashes"]:
        verdicts.append("EXPECTED_HASHES_MISSING")
    if "docs/research/STABLE_LOOP_PHASE_LOCK_051_ARTIFACT_CHECKLIST.md" in failures["missing_files"]:
        verdicts.append("ARTIFACT_CHECKLIST_MISSING")
    if any("STABLE_LOOP_PHASE_LOCK_051_TABLES.md" in item for item in failures["missing_sections"]):
        verdicts.append("PAPER_TABLE_SOURCE_MISSING")
    if not prod_ok:
        verdicts.append("CLAIM_BOUNDARY_MISSING")
    return list(dict.fromkeys(verdicts))


def run_check() -> dict[str, Any]:
    missing_files = missing_required_files()
    docs: dict[str, str] = {}
    for key, rel_path in REQUIRED_DOCS.items():
        path = ROOT / rel_path
        docs[key] = read_text(path) if path.exists() else ""
    bundle_text = "\n".join(docs.values())

    invalid_docs, missing_sections = validate_docs(docs)
    missing_commands = validate_commands(docs.get("readme", ""))
    missing_hashes = validate_hashes(bundle_text) if EXPECTED_HASHES_PATH.exists() else ["expected_hashes_file"]
    missing_references = validate_references(bundle_text)
    missing_verdicts = validate_verdicts(bundle_text)
    prod_ok = production_flags_ok(bundle_text)

    if invalid_docs:
        missing_sections.extend(invalid_docs)
    for reference in missing_references:
        missing_sections.append(f"missing_reference:{reference}")
    for verdict in missing_verdicts:
        missing_sections.append(f"missing_verdict:{verdict}")

    failures = {
        "missing_files": missing_files,
        "missing_sections": missing_sections,
        "missing_commands": missing_commands,
        "missing_hashes": missing_hashes,
    }
    check_pass = (
        not missing_files
        and not missing_sections
        and not missing_commands
        and not missing_hashes
        and prod_ok
    )
    return {
        "probe": PROBE,
        "check_pass": check_pass,
        "missing_files": missing_files,
        "missing_sections": missing_sections,
        "missing_commands": missing_commands,
        "missing_hashes": missing_hashes,
        "verdicts": derive_verdicts(check_pass, failures, prod_ok),
        "production_flags_ok": prod_ok,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.check_only:
        print("Use --check-only for static reviewer bundle validation.", file=sys.stderr)
        return 2
    result = run_check()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
