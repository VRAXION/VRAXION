#!/usr/bin/env python3
"""Checker for E83 CALC-SCRIBE v003 Local Golden reload artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = [
    "run_manifest.json",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "seed_results.json",
    "governance_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "report.md",
    "pocket_library/calc_scribe_v003/registry.json",
    "pocket_library/calc_scribe_v003/tokens.json",
    "pocket_library/calc_scribe_v003/artifacts/calc_scribe_v003.json",
    "pocket_library/calc_scribe_v003/promotion_ledger.jsonl",
    "pocket_library/calc_scribe_v003/score_ledger.jsonl",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e83_calc_scribe_v003_local_golden_promotion_reload")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    failures: list[str] = []
    for name in REQUIRED:
        if not (out / name).exists():
            failures.append(f"missing artifact: {name}")
    if not failures:
        manifest = read_json(out / "run_manifest.json")
        aggregate = read_json(out / "aggregate_metrics.json")
        governance = read_json(out / "governance_report.json")
        decision = read_json(out / "decision.json")
        artifact = read_json(out / "pocket_library/calc_scribe_v003/artifacts/calc_scribe_v003.json")
        progress = [line for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if manifest.get("artifact_contract") != "E83_CALC_SCRIBE_V003_LOCAL_GOLDEN_PROMOTION_RELOAD":
            failures.append("artifact contract mismatch")
        if manifest.get("scope") != "visible_calc_trace_validator":
            failures.append("scope mismatch")
        if artifact.get("scope") != "visible_calc_trace_validator":
            failures.append("artifact scope mismatch")
        if "gsm8k_solver" not in artifact.get("not_claims", []):
            failures.append("missing not_claims gsm8k_solver")
        if len(progress) < 3:
            failures.append("progress writeout too sparse")
        if not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing heartbeat")
        if decision.get("failure_count") != 0:
            failures.append("decision failure_count != 0")
        if aggregate.get("reload_match_rate") != 1.0:
            failures.append("reload mismatch")
        if aggregate.get("validation", {}).get("marker_min") != 1.0:
            failures.append("validation marker min not 1.0")
        if aggregate.get("adversarial", {}).get("action_min") != 1.0:
            failures.append("adversarial action min not 1.0")
        for key in ["load_ok", "tamper_blocked", "token_swap_blocked", "redundant_clone_blocked", "unsafe_global_scope_blocked"]:
            if governance.get(key) is not True:
                failures.append(f"governance gate failed: {key}")
        if governance.get("bad_promotion_count") != 0:
            failures.append("bad promotion count != 0")
    summary = {
        "checker": "E83_CALC_SCRIBE_V003_LOCAL_GOLDEN_PROMOTION_RELOAD_CHECK",
        "out": str(out),
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }
    if args.write_summary:
        (out / "checker_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
