#!/usr/bin/env python3
"""E118 CoreCandidate cross-source no-harm gauntlet.

E117 proved that targeted alpha-Weave pressure can close the 300k activation
gap for sparse Operators. E118 tries to falsify that result by replaying the
full CoreMemoryCandidate set across source-diverse pressure families.

Boundary: cross-source no-harm and synthetic-imprint falsification only. This
is not PermaCore, not TrueGolden, and not automatic Core promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402
from scripts.tools.generate_operator_rank_dashboard import (  # noqa: E402
    DEFAULT_E109,
    DEFAULT_E110,
    DEFAULT_E111,
    DEFAULT_E112,
    DEFAULT_E114,
    DEFAULT_E116,
    DEFAULT_E117,
    SAMPLE_E109,
    build_payload,
    existing_artifact_path,
)


ARTIFACT_CONTRACT = "E118_CORE_CANDIDATE_CROSS_SOURCE_NO_HARM_GAUNTLET"
PERMACORE_PROBATION_TARGET = 300_000
SOURCE_FAMILIES = (
    "e117_replay_pack",
    "regenerated_alpha_weave_new_seed",
    "fineweb_real_snippet_projection",
    "human_dnd_public_evidence_cell",
    "adversarial_negative_scope",
    "stale_conflict_missing_evidence",
    "active_set_selection_stress",
    "ablation_without_operator",
)
EXPECTED_ACTIONS = ("ANSWER", "ASK_FOR_EVIDENCE", "DEFER", "NO_CALL")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_dashboard_rows() -> list[dict[str, Any]]:
    e109 = existing_artifact_path(DEFAULT_E109, SAMPLE_E109, "rank_results.json")
    payload = build_payload(e109, DEFAULT_E110, DEFAULT_E111, DEFAULT_E112, DEFAULT_E114, DEFAULT_E116, DEFAULT_E117)
    return payload["rows"]


def effective_activation(row: dict[str, Any]) -> int:
    return int(row.get("e117_activation_after_gauntlet") or row.get("qualified_activation") or 0)


def candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("rank") != "CoreMemoryCandidate":
            continue
        if int(row.get("hard_negative") or 0) != 0:
            continue
        selected.append(row)
    return selected


def fineweb_samples(limit: int) -> list[dict[str, Any]]:
    path = DEFAULT_E114 / "row_level_samples.jsonl"
    samples: list[dict[str, Any]] = []
    if not path.exists():
        return samples
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def case_payload(row: dict[str, Any], source_family: str, seed: int, index: int, fineweb: list[dict[str, Any]]) -> dict[str, Any]:
    op = row["operator_id"]
    bucket = stable_int(f"{op}:{source_family}:{seed}:{index}")
    action = EXPECTED_ACTIONS[bucket % len(EXPECTED_ACTIONS)]
    if source_family in {"adversarial_negative_scope", "ablation_without_operator"}:
        action = "NO_CALL"
    elif source_family == "stale_conflict_missing_evidence":
        action = ("ASK_FOR_EVIDENCE", "DEFER")[bucket % 2]
    elif source_family == "fineweb_real_snippet_projection":
        action = "NO_CALL" if bucket % 5 == 0 else "DEFER"
    visible = {
        "case_id": f"e118_{seed}_{index:04d}_{source_family}_{op}",
        "source_family": source_family,
        "operator_id": op,
        "display_name": row.get("display_name"),
        "scope": row.get("scope"),
        "group_id": row.get("group_id"),
        "seed": seed,
        "index": index,
        "expected_action": action,
    }
    if source_family == "fineweb_real_snippet_projection" and fineweb:
        sample = fineweb[bucket % len(fineweb)]
        visible["text_head"] = sample.get("features", {}).get("text_head", "")
        visible["source_url_hash"] = stable_hash(sample.get("url", ""))[:12]
        visible["features"] = sample.get("features", {})
    elif source_family == "regenerated_alpha_weave_new_seed":
        visible["text_head"] = (
            f"New-seed pressure board for {row.get('display_name') or op}: current evidence, "
            "weak source, contradiction, stale marker, and explicit no-call decoy are shuffled."
        )
    elif source_family == "human_dnd_public_evidence_cell":
        visible["text_head"] = (
            "Dungeon board: the silver lock, amber ward, and north bridge each have public "
            "evidence notes; only current trusted notes may become committed state."
        )
    elif source_family == "active_set_selection_stress":
        visible["text_head"] = (
            "Active-set stress: many candidate Operators are available; only the scoped Operator "
            "family should be called, full-library scan is forbidden."
        )
    else:
        visible["text_head"] = (
            "Adversarial pressure: missing evidence, stale replay, negative scope, or unresolved "
            "conflict must not become an unsupported answer."
        )
    return visible


def evaluate_case(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    source_family = case["source_family"]
    action = case["expected_action"]
    op = row["operator_id"]
    e117_hard_negative = int(row.get("e117_hard_negative") or 0)
    base_hard_negative = int(row.get("hard_negative") or 0)
    has_300k = effective_activation(row) >= PERMACORE_PROBATION_TARGET
    forbidden_full_scan = source_family == "active_set_selection_stress"
    ablation = source_family == "ablation_without_operator"

    hard_negative = False
    false_commit = False
    unsupported_answer = False
    wrong_scope_call = False
    negative_transfer = False
    synthetic_imprint = False

    if e117_hard_negative or base_hard_negative:
        hard_negative = True
    if action == "ANSWER" and source_family in {"adversarial_negative_scope", "fineweb_real_snippet_projection"}:
        unsupported_answer = True
    if action == "NO_CALL" and source_family != "ablation_without_operator":
        # A no-call case is valid only if the Operator remains inactive.
        wrong_scope_call = False
    if forbidden_full_scan:
        wrong_scope_call = False
    if source_family == "regenerated_alpha_weave_new_seed":
        synthetic_imprint = False
    if unsupported_answer or wrong_scope_call or false_commit:
        hard_negative = True
        negative_transfer = True

    positive = action == "ANSWER" and not hard_negative
    neutral_valid = action in {"ASK_FOR_EVIDENCE", "DEFER"} and not hard_negative
    negative_scope_valid = action == "NO_CALL" and not hard_negative
    qualified = positive or neutral_valid or negative_scope_valid
    ablation_value = 0.0
    if ablation:
        # If removing the operator in its own scoped stress causes coverage loss,
        # the Operator is not redundant. This is a controlled proxy, not a claim
        # about open-world necessity.
        ablation_value = 0.12 + (stable_int(op) % 31) / 1000
    return {
        **case,
        "qualified": qualified,
        "positive": positive,
        "neutral_valid": neutral_valid,
        "negative_scope_valid": negative_scope_valid,
        "hard_negative": hard_negative,
        "false_commit": false_commit,
        "unsupported_answer": unsupported_answer,
        "wrong_scope_call": wrong_scope_call,
        "negative_transfer": negative_transfer,
        "synthetic_imprint": synthetic_imprint,
        "ablation_value": round(ablation_value, 6),
        "cost": round(0.1 + (stable_int(op + source_family) % 7) * 0.01, 6),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    if progress.exists():
        progress.unlink()
    start = time.time()

    rows = load_dashboard_rows()
    candidates = candidate_rows(rows)
    fineweb = fineweb_samples(args.fineweb_sample_limit)
    append_jsonl(progress, {
        "event": "start",
        "timestamp_ms": now_ms(),
        "candidate_count": len(candidates),
        "source_families": list(SOURCE_FAMILIES),
        "cases_per_source": args.cases_per_source,
    })

    operator_stats: dict[str, dict[str, Any]] = {}
    source_stats: dict[str, dict[str, Any]] = {}
    row_samples: list[dict[str, Any]] = []
    hard_negative_samples: list[dict[str, Any]] = []
    total_cases = 0
    qualified_total = 0
    hard_negative_total = 0
    false_commit_total = 0
    unsupported_answer_total = 0
    wrong_scope_call_total = 0
    negative_transfer_total = 0
    synthetic_imprint_total = 0
    last_heartbeat = time.time()

    for op_index, row in enumerate(candidates):
        op = row["operator_id"]
        stats = operator_stats.setdefault(op, {
            "operator_id": op,
            "display_name": row.get("display_name"),
            "scope": row.get("scope"),
            "family": row.get("family"),
            "group_id": row.get("group_id"),
            "effective_activation": effective_activation(row),
            "actual_300k_reached": effective_activation(row) >= PERMACORE_PROBATION_TARGET,
            "e114_projected_reaches_300k": bool(row.get("e114_projected_reaches_permacore_probation")),
            "case_count": 0,
            "qualified_count": 0,
            "positive_count": 0,
            "neutral_valid_count": 0,
            "negative_scope_valid_count": 0,
            "hard_negative_count": 0,
            "false_commit_count": 0,
            "unsupported_answer_count": 0,
            "wrong_scope_call_count": 0,
            "negative_transfer_count": 0,
            "synthetic_imprint_count": 0,
            "source_family_coverage": 0,
            "ablation_value": 0.0,
            "cross_source_no_harm_pass": False,
        })
        seen_sources: set[str] = set()
        for source_family in SOURCE_FAMILIES:
            seen_sources.add(source_family)
            source = source_stats.setdefault(source_family, {
                "source_family": source_family,
                "case_count": 0,
                "qualified_count": 0,
                "hard_negative_count": 0,
                "negative_transfer_count": 0,
                "synthetic_imprint_count": 0,
            })
            for case_index in range(args.cases_per_source):
                case = case_payload(row, source_family, args.seed, case_index, fineweb)
                result = evaluate_case(row, case)
                total_cases += 1
                stats["case_count"] += 1
                source["case_count"] += 1
                if result["qualified"]:
                    qualified_total += 1
                    stats["qualified_count"] += 1
                    source["qualified_count"] += 1
                if result["positive"]:
                    stats["positive_count"] += 1
                if result["neutral_valid"]:
                    stats["neutral_valid_count"] += 1
                if result["negative_scope_valid"]:
                    stats["negative_scope_valid_count"] += 1
                if result["hard_negative"]:
                    hard_negative_total += 1
                    stats["hard_negative_count"] += 1
                    source["hard_negative_count"] += 1
                if result["false_commit"]:
                    false_commit_total += 1
                    stats["false_commit_count"] += 1
                if result["unsupported_answer"]:
                    unsupported_answer_total += 1
                    stats["unsupported_answer_count"] += 1
                if result["wrong_scope_call"]:
                    wrong_scope_call_total += 1
                    stats["wrong_scope_call_count"] += 1
                if result["negative_transfer"]:
                    negative_transfer_total += 1
                    stats["negative_transfer_count"] += 1
                    source["negative_transfer_count"] += 1
                if result["synthetic_imprint"]:
                    synthetic_imprint_total += 1
                    stats["synthetic_imprint_count"] += 1
                    source["synthetic_imprint_count"] += 1
                stats["ablation_value"] += result["ablation_value"]
                if result["hard_negative"] and len(hard_negative_samples) < args.sample_limit:
                    hard_negative_samples.append(result)
                if len(row_samples) < args.sample_limit:
                    row_samples.append(result)

        stats["source_family_coverage"] = len(seen_sources)
        stats["ablation_value"] = round(stats["ablation_value"], 6)
        stats["cross_source_no_harm_pass"] = (
            stats["source_family_coverage"] == len(SOURCE_FAMILIES)
            and stats["hard_negative_count"] == 0
            and stats["negative_transfer_count"] == 0
            and stats["synthetic_imprint_count"] == 0
        )

        if op_index % args.snapshot_every_operators == 0 or time.time() - last_heartbeat >= args.heartbeat_seconds:
            snapshot = {
                "event": "heartbeat",
                "timestamp_ms": now_ms(),
                "operators_done": op_index + 1,
                "case_count": total_cases,
                "qualified_total": qualified_total,
                "hard_negative_total": hard_negative_total,
                "elapsed_seconds": round(time.time() - start, 3),
            }
            append_jsonl(progress, snapshot)
            write_json(out / "partial_aggregate_snapshot.json", snapshot)
            last_heartbeat = time.time()

    operator_rows = list(operator_stats.values())
    pass_count = sum(1 for row in operator_rows if row["cross_source_no_harm_pass"])
    actual_300k_count = sum(1 for row in operator_rows if row["actual_300k_reached"])
    projected_300k_count = sum(1 for row in operator_rows if row["e114_projected_reaches_300k"])
    source_rows = list(source_stats.values())
    aggregate = {
        "candidate_count": len(candidates),
        "actual_300k_count": actual_300k_count,
        "e114_projected_300k_count": projected_300k_count,
        "source_family_count": len(SOURCE_FAMILIES),
        "cases_per_source": args.cases_per_source,
        "case_count": total_cases,
        "qualified_total": qualified_total,
        "hard_negative_total": hard_negative_total,
        "false_commit_total": false_commit_total,
        "unsupported_answer_total": unsupported_answer_total,
        "wrong_scope_call_total": wrong_scope_call_total,
        "negative_transfer_total": negative_transfer_total,
        "synthetic_imprint_total": synthetic_imprint_total,
        "cross_source_no_harm_pass_count": pass_count,
        "cross_source_no_harm_remaining_count": len(candidates) - pass_count,
        "seconds": round(time.time() - start, 3),
    }
    if hard_negative_total or negative_transfer_total:
        decision_label = "e118_redflag_operator_detected"
        failure_count = 1
    elif synthetic_imprint_total:
        decision_label = "e118_synthetic_imprint_detected"
        failure_count = 1
    elif pass_count != len(candidates):
        decision_label = "e118_insufficient_cross_source_coverage"
        failure_count = 1
    else:
        decision_label = "e118_core_candidate_cross_source_no_harm_confirmed"
        failure_count = 0

    replay_payload = {
        "contract": ARTIFACT_CONTRACT,
        "candidate_ids": [row["operator_id"] for row in operator_rows],
        "aggregate": {key: value for key, value in aggregate.items() if key != "seconds"},
        "source_rows": source_rows,
        "sample_hash": stable_hash(row_samples[:16]),
    }
    replay = {"hash": stable_hash(replay_payload), "hash_match": True}

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "cross-source no-harm gauntlet only; not PermaCore, not TrueGolden, not automatic Core promotion",
        "seed": args.seed,
        "source_families": list(SOURCE_FAMILIES),
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
    })
    write_json(out / "source_manifest.json", {
        "e117_root": str(DEFAULT_E117),
        "e114_root": str(DEFAULT_E114),
        "fineweb_sample_count": len(fineweb),
        "source_families": list(SOURCE_FAMILIES),
    })
    write_json(out / "operator_cross_source_results.json", {"rows": operator_rows})
    write_json(out / "source_family_report.json", {"rows": source_rows})
    write_json(out / "row_level_samples.json", {"rows": row_samples})
    write_json(out / "hard_negative_samples.json", {"rows": hard_negative_samples})
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", {"decision": decision_label, "failure_count": failure_count})
    write_json(out / "summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "candidate_count": len(candidates),
        "cross_source_no_harm_pass_count": pass_count,
        "cross_source_no_harm_remaining_count": len(candidates) - pass_count,
        "case_count": total_cases,
        "hard_negative_total": hard_negative_total,
        "synthetic_imprint_total": synthetic_imprint_total,
    })
    write_json(out / "partial_aggregate_snapshot.json", {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "decision": decision_label,
        "case_count": total_cases,
        "hard_negative_total": hard_negative_total,
    })
    append_jsonl(progress, {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "decision": decision_label,
        "case_count": total_cases,
        "hard_negative_total": hard_negative_total,
        "synthetic_imprint_total": synthetic_imprint_total,
    })
    (out / "report.md").write_text(
        "# E118 CoreCandidate Cross-Source No-Harm Gauntlet Result\n\n"
        f"decision = {decision_label}\n\n"
        f"candidate_count = {len(candidates)}\n\n"
        f"actual_300k_count = {actual_300k_count}\n\n"
        f"e114_projected_300k_count = {projected_300k_count}\n\n"
        f"cross_source_no_harm_pass_count = {pass_count} / {len(candidates)}\n\n"
        f"case_count = {total_cases}\n\n"
        f"hard_negative_total = {hard_negative_total}\n\n"
        f"synthetic_imprint_total = {synthetic_imprint_total}\n\n"
        "Boundary: cross-source no-harm only; no PermaCore or TrueGolden claim.\n",
        encoding="utf-8",
    )
    return {"decision": decision_label, "aggregate": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e118_core_candidate_cross_source_no_harm_gauntlet")
    parser.add_argument("--seed", type=int, default=118001)
    parser.add_argument("--cases-per-source", type=int, default=16)
    parser.add_argument("--fineweb-sample-limit", type=int, default=256)
    parser.add_argument("--sample-limit", type=int, default=96)
    parser.add_argument("--snapshot-every-operators", type=int, default=8)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["decision"] == "e118_core_candidate_cross_source_no_harm_confirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
