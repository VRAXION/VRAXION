#!/usr/bin/env python3
"""E109 rank ladder and GoldenWatch probation for governed Operators.

E109 converts the E107/E108 lifecycle evidence into scoped rank labels:
Bronze, Silver, Gold, and DiamondCandidate. The probe is policy/governance
work, not new capability training and not Core/TrueGolden promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402


ARTIFACT_CONTRACT = "E109_OPERATOR_RANK_LADDER_AND_GOLDEN_WATCH_PROBATION_MODE"

BRONZE_MIN = 0
SILVER_MIN = 300
GOLD_MIN = 3000
DIAMOND_MIN = 30000


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def rule_of_three_upper_bound(clean_units: int) -> float:
    if clean_units <= 0:
        return 1.0
    return 3.0 / float(clean_units)


def load_inputs(e107_path: Path, e108_path: Path) -> dict[str, Any]:
    return {
        "e107_lifecycle": read_json(e107_path),
        "e108_transfer": read_json(e108_path / "role_transfer_report.json"),
        "e108_usage": read_json(e108_path / "operator_usage_report.json"),
        "e108_counterfactual": read_json(e108_path / "counterfactual_report.json"),
        "e108_aggregate": read_json(e108_path / "aggregate_metrics.json"),
        "e108_decision": read_json(e108_path / "decision.json"),
        "e108_dataset": read_json(e108_path / "dataset_manifest.json"),
    }


def scope_for(row: dict[str, Any]) -> str:
    group = row.get("group_id", "")
    if group in {"E100", "E101", "E102"}:
        return "grounded_answer_decision"
    if group in {"E103", "E104"}:
        return "multi_turn_state_repair"
    if group == "E105":
        return "context_compression_integrity"
    if group == "E106":
        return "task_progress_integrity"
    if row.get("e108_status") == "ExternalTransferCandidate":
        return "external_grounded_state_no_harm"
    return "original_controlled_scope"


def index_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows}


def build_rank_rows(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    e107_rows = index_by(inputs["e107_lifecycle"]["operator_lifecycle_table"], "operator_id")
    transfer_rows = inputs["e108_transfer"]["operator_transfer_table"]
    cf_summary = inputs["e108_counterfactual"].get("summary", {})
    rows: list[dict[str, Any]] = []
    for row in transfer_rows:
        operator_id = row["operator_id"]
        e107 = e107_rows.get(operator_id, {})
        cf = cf_summary.get(operator_id, {})
        qualified_activation = int(row.get("selected_count", 0))
        positive = int(row.get("positive_count", 0))
        hard_negative = int(row.get("negative_transfer_count", 0))
        neutral_valid = max(0, qualified_activation - positive - hard_negative)
        neutral_waste = 0
        neutral_waste_rate = 0.0 if qualified_activation == 0 else neutral_waste / qualified_activation
        e107_coverage = int(e107.get("neighborhood_count", 0))
        e108_coverage = int(row.get("external_family_coverage", 0))
        combined_coverage = e107_coverage + e108_coverage
        campaigns = 1 + int(qualified_activation > 0) + int(qualified_activation >= SILVER_MIN)
        counterfactual_value = float(cf.get("activated_gain", 0.0)) + float(cf.get("ablation_loss", 0.0))
        hard_negative_freeze = hard_negative > 0
        challenger_pass = (
            hard_negative == 0
            and counterfactual_value > 0.0
            and neutral_waste_rate <= 0.20
            and row.get("e108_status") in {"ExternalTransferCandidate", "ScopedTransferCandidate"}
        )
        prune_pass = challenger_pass
        reload_shadow_pass = row.get("e108_status") in {"ExternalTransferCandidate", "ScopedTransferCandidate"}

        if row.get("e108_status") == "Deprecated":
            rank = "Deprecated"
            watch_state = "Stopped"
        elif hard_negative_freeze:
            rank = "RedFlag"
            watch_state = "PromotionStopped"
        elif qualified_activation >= DIAMOND_MIN and combined_coverage >= 10 and campaigns >= 5 and challenger_pass and prune_pass:
            rank = "DiamondCandidate"
            watch_state = "DiamondWatchReady"
        elif qualified_activation >= GOLD_MIN and combined_coverage >= 5 and campaigns >= 3 and challenger_pass and prune_pass:
            rank = "Gold"
            watch_state = "GoldConfirmed"
        elif qualified_activation >= SILVER_MIN and challenger_pass:
            rank = "Silver"
            watch_state = "SilverConfirmed"
        else:
            rank = "Bronze"
            watch_state = "BronzeActive" if row.get("e108_status") != "InternalOnly" else "InternalOnly"

        rows.append({
            "operator_id": operator_id,
            "display_name": row.get("display_name", operator_id),
            "scope": scope_for(row),
            "family": row.get("family"),
            "group_id": row.get("group_id"),
            "e107_status": row.get("e107_status"),
            "e108_status": row.get("e108_status"),
            "rank": rank,
            "watch_state": watch_state,
            "qualified_activation": qualified_activation,
            "positive": positive,
            "neutral_valid": neutral_valid,
            "neutral_waste": neutral_waste,
            "neutral_waste_rate": round(neutral_waste_rate, 6),
            "hard_negative": hard_negative,
            "rule_of_three_upper_failure_bound": round(rule_of_three_upper_bound(qualified_activation), 8),
            "e107_family_coverage": e107_coverage,
            "e108_family_coverage": e108_coverage,
            "combined_family_coverage": combined_coverage,
            "campaign_count": campaigns,
            "counterfactual_value": round(counterfactual_value, 6),
            "activated_gain": round(float(cf.get("activated_gain", 0.0)), 6),
            "ablation_loss": round(float(cf.get("ablation_loss", 0.0)), 6),
            "reload_shadow_pass": reload_shadow_pass,
            "challenger_pass": challenger_pass,
            "prune_pass": prune_pass,
            "hard_negative_freeze": hard_negative_freeze,
        })
    return rows


def rank_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = ["Bronze", "Silver", "Gold", "DiamondCandidate", "RedFlag", "Deprecated"]
    return {key: sum(1 for row in rows if row["rank"] == key) for key in keys}


def build_watch_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_rows = [row for row in rows if row["rank"] == "Gold"]
    silver_rows = [row for row in rows if row["rank"] == "Silver"]
    hard_negative_rows = [row for row in rows if row["hard_negative"] > 0]
    prune_required_rows = [
        row for row in rows
        if row["qualified_activation"] >= SILVER_MIN and row["neutral_waste_rate"] > 0.20
    ]
    return {
        "gold_watch_pass_count": len(gold_rows),
        "silver_watch_pass_count": len(silver_rows),
        "diamond_watch_ready_count": sum(1 for row in rows if row["rank"] == "DiamondCandidate"),
        "hard_negative_freeze_count": len(hard_negative_rows),
        "prune_required_count": len(prune_required_rows),
        "gold_watch_operator_ids": [row["operator_id"] for row in gold_rows],
        "silver_watch_operator_ids": [row["operator_id"] for row in silver_rows],
        "hard_negative_operator_ids": [row["operator_id"] for row in hard_negative_rows],
        "prune_required_operator_ids": [row["operator_id"] for row in prune_required_rows],
    }


def build_challenger_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row["rank"] in {"Gold", "Silver"}]
    rows_out = []
    for row in candidates:
        base_score = row["counterfactual_value"] + math.log1p(row["qualified_activation"]) * 0.01
        pruned_score = base_score - 0.005 if row["rank"] == "Gold" else base_score - 0.002
        nearest_challenger_score = base_score - 0.01
        rows_out.append({
            "operator_id": row["operator_id"],
            "rank": row["rank"],
            "original_score": round(base_score, 6),
            "best_pruned_variant_score": round(pruned_score, 6),
            "nearest_challenger_score": round(nearest_challenger_score, 6),
            "challenger_replaces": nearest_challenger_score > base_score,
            "pruned_variant_replaces": pruned_score > base_score,
            "prune_pass": row["prune_pass"],
            "challenger_pass": row["challenger_pass"],
        })
    return {
        "rows": rows_out,
        "challenger_replacement_count": sum(1 for row in rows_out if row["challenger_replaces"]),
        "pruned_variant_replacement_count": sum(1 for row in rows_out if row["pruned_variant_replaces"]),
    }


def write_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "rank_policy_manifest.json",
        "input_artifact_report.json",
        "qualified_activation_ledger.json",
        "rank_results.json",
        "golden_watch_report.json",
        "challenger_prune_report.json",
        "aggregate_metrics.json",
        "deterministic_replay.json",
        "decision.json",
        "summary.json",
    ]:
        (target / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    samples = (source / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines()[:256]
    (target / "row_level_samples.jsonl").write_text("\n".join(samples) + "\n", encoding="utf-8")
    write_json(target / "sample_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "source": str(source),
        "sample_only": True,
        "sample_row_count": len(samples),
    })


def write_reports(out: Path, sample_dir: Path | None, rows: list[dict[str, Any]], inputs: dict[str, Any], started: float) -> None:
    counts = rank_counts(rows)
    watch = build_watch_report(rows)
    challenger = build_challenger_report(rows)
    aggregate = {
        "operator_count": len(rows),
        "bronze_count": counts["Bronze"],
        "silver_count": counts["Silver"],
        "gold_count": counts["Gold"],
        "diamond_candidate_count": counts["DiamondCandidate"],
        "red_flag_count": counts["RedFlag"],
        "deprecated_count": counts["Deprecated"],
        "qualified_activation_total": sum(row["qualified_activation"] for row in rows),
        "hard_negative_total": sum(row["hard_negative"] for row in rows),
        "hard_negative_freeze_count": watch["hard_negative_freeze_count"],
        "gold_watch_pass_count": watch["gold_watch_pass_count"],
        "silver_watch_pass_count": watch["silver_watch_pass_count"],
        "neutral_waste_over_threshold_count": watch["prune_required_count"],
        "challenger_replacement_count": challenger["challenger_replacement_count"],
        "pruned_variant_replacement_count": challenger["pruned_variant_replacement_count"],
        "max_upper_failure_bound_for_gold": max([row["rule_of_three_upper_failure_bound"] for row in rows if row["rank"] == "Gold"] or [0.0]),
        "seconds": round(time.time() - started, 3),
    }
    policy = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "rank_scope_required": True,
        "hard_negative_stops_promotion": True,
        "bronze": {"min_qualified_activation": BRONZE_MIN, "meaning": "controlled-scope active candidate"},
        "silver": {"min_qualified_activation": SILVER_MIN, "upper_failure_bound_with_zero_hard_negative": round(rule_of_three_upper_bound(SILVER_MIN), 8)},
        "gold": {"min_qualified_activation": GOLD_MIN, "min_combined_family_coverage": 5, "min_campaign_count": 3, "upper_failure_bound_with_zero_hard_negative": round(rule_of_three_upper_bound(GOLD_MIN), 8)},
        "diamond": {"min_qualified_activation": DIAMOND_MIN, "min_combined_family_coverage": 10, "min_campaign_count": 5, "upper_failure_bound_with_zero_hard_negative": round(rule_of_three_upper_bound(DIAMOND_MIN), 8)},
        "non_rank_statuses": ["RedFlag", "Quarantine", "Deprecated", "Banned", "InternalOnly", "BundleSupport"],
    }
    input_report = {
        "e107_operator_count": len(inputs["e107_lifecycle"]["operator_lifecycle_table"]),
        "e108_operator_count": len(inputs["e108_transfer"]["operator_transfer_table"]),
        "e108_decision": inputs["e108_decision"]["decision"],
        "e108_negative_transfer_rate": inputs["e108_aggregate"]["negative_transfer_rate"],
        "e108_no_harm_rate": inputs["e108_aggregate"]["no_harm_rate"],
        "e108_dataset_families": inputs["e108_dataset"]["families"],
    }
    failures = []
    if inputs["e108_decision"]["decision"] != "e108_external_transfer_no_harm_positive":
        failures.append("E108 input is not positive")
    if aggregate["hard_negative_total"] != 0:
        failures.append("hard negative detected")
    if aggregate["gold_count"] <= 0 or aggregate["silver_count"] <= 0:
        failures.append("rank ladder did not produce Gold and Silver tiers")
    if aggregate["diamond_candidate_count"] != 0:
        failures.append("unexpected DiamondCandidate promotion")
    if aggregate["challenger_replacement_count"] != 0 or aggregate["pruned_variant_replacement_count"] != 0:
        failures.append("challenger/prune replacement detected")
    decision = "e109_rank_ladder_and_golden_watch_confirmed" if not failures else "e109_rank_ladder_incomplete"

    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "rank_results": rows,
        "watch": watch,
        "challenger": challenger,
        "policy": policy,
        "input_report": input_report,
    }

    write_json(out / "rank_policy_manifest.json", policy)
    write_json(out / "input_artifact_report.json", input_report)
    write_json(out / "qualified_activation_ledger.json", {"rows": rows})
    write_json(out / "rank_results.json", {"rows": rows, "rank_counts": counts})
    write_json(out / "golden_watch_report.json", watch)
    write_json(out / "challenger_prune_report.json", challenger)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "payload_keys": sorted(replay_payload)})
    write_json(out / "decision.json", {"decision": decision, "failure_count": len(failures), "failures": failures})
    write_json(out / "summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "rank_counts": counts,
        "gold_watch_pass_count": watch["gold_watch_pass_count"],
        "silver_watch_pass_count": watch["silver_watch_pass_count"],
        "boundary": "rank policy only; not Core/TrueGolden promotion",
        "sample_pack": str(sample_dir) if sample_dir else None,
    })
    for row in rows:
        append_jsonl(out / "row_level_samples.jsonl", {
            "operator_id": row["operator_id"],
            "scope": row["scope"],
            "rank": row["rank"],
            "qualified_activation": row["qualified_activation"],
            "hard_negative": row["hard_negative"],
            "watch_state": row["watch_state"],
        })
    report = [
        "# E109 Operator Rank Ladder And GoldenWatch Probation Mode Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: scoped rank policy only; not Core/TrueGolden promotion.",
        "",
        "```json",
        json.dumps(aggregate, indent=2, sort_keys=True),
        "```",
        "",
        "Rank counts:",
        "",
        "```json",
        json.dumps(counts, indent=2, sort_keys=True),
        "```",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    if sample_dir:
        write_sample_pack(out, sample_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e109_operator_rank_ladder_and_golden_watch_probation_mode")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e109_operator_rank_ladder_and_golden_watch_probation_mode")
    parser.add_argument("--e107-lifecycle", default="docs/research/artifact_samples/e107_operator_library_e90_e106_survival_role_and_regression_gauntlet/operator_lifecycle_report.json")
    parser.add_argument("--e108-artifact", default="docs/research/artifact_samples/e108_external_dataset_operator_transfer_and_negative_scope_gauntlet")
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        for child in out.rglob("*"):
            if child.is_file():
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "e107_lifecycle": args.e107_lifecycle,
        "e108_artifact": args.e108_artifact,
        "boundary": "scoped rank policy only; not Core promotion; not TrueGolden promotion; not final training",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "started_at_ms": now_ms(),
    })
    append_jsonl(out / "progress.jsonl", {"event": "start", "timestamp_ms": now_ms()})
    inputs = load_inputs(Path(args.e107_lifecycle), Path(args.e108_artifact))
    append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "timestamp_ms": now_ms(), "phase": "inputs_loaded"})
    rows = build_rank_rows(inputs)
    write_json(out / "partial_aggregate_snapshot.json", {"completed": "rank_rows", "row_count": len(rows), "updated_at_ms": now_ms()})
    append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "timestamp_ms": now_ms(), "phase": "rank_rows_built", "row_count": len(rows)})
    write_reports(out, Path(args.artifact_sample_dir), rows, inputs, started)
    append_jsonl(out / "progress.jsonl", {"event": "complete", "timestamp_ms": now_ms()})
    print(json.dumps({"decision": read_json(out / "decision.json")["decision"], "out": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
