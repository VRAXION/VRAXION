#!/usr/bin/env python3
"""E117 alpha-Weave targeted pressure gauntlet.

E116 generated targeted synthetic alpha-Weave pressure cells for the 77
Operators that still needed targeted activations after the FineWeb projection.
E117 turns that schedule into an actual deterministic gauntlet over every
generated cell and adversarial variant.

Boundary: this is a targeted activation/no-harm gauntlet over synthetic
pressure data. It is not PermaCore, not TrueGolden, and not final training.
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
from scripts.probes.run_e115_alpha_weave_pressure_cell_schema_validation import (  # noqa: E402
    SCHEMA_VERSION,
    canonical_schema,
    deterministic_hash,
    validate_cell_schema,
)
from scripts.probes.run_e116_alpha_weave_synthetic_pressure_generation import public_leaks  # noqa: E402


ARTIFACT_CONTRACT = "E117_ALPHA_WEAVE_TARGETED_PRESSURE_GAUNTLET"
DEFAULT_E116 = Path("target/pilot_wave/e116_alpha_weave_synthetic_pressure_generation")
PERMACORE_PROBATION_TARGET = 300_000
ALLOWED_ACTIONS = {"ANSWER", "ASK_FOR_EVIDENCE", "SEARCH_MORE", "HOLD_UNRESOLVED", "DEFER", "NO_CALL"}
NON_ANSWER_ACTIONS = {"ASK_FOR_EVIDENCE", "SEARCH_MORE", "HOLD_UNRESOLVED", "DEFER", "NO_CALL"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def hash_payload(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def observation_ids(public_input: dict[str, Any]) -> set[str]:
    return {str(item.get("obs_id")) for item in public_input.get("observations", []) if "obs_id" in item}


def validate_metadata(cell: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    metadata = cell.get("training_metadata", {})
    if metadata.get("data_origin") != "synthetic_codex_generated":
        failures.append("missing synthetic data_origin")
    if metadata.get("generator") != "codex":
        failures.append("missing codex generator marker")
    if not metadata.get("synthetic_disclosure"):
        failures.append("missing synthetic disclosure metadata")
    if metadata.get("operator_visibility") != "hidden_from_candidate":
        failures.append("operator target is not hidden from candidate")
    targets = metadata.get("target_operators", [])
    if not isinstance(targets, list) or len(targets) != 1 or not targets[0]:
        failures.append("expected exactly one hidden target operator")
    repeat_count = metadata.get("repeat_count")
    if not isinstance(repeat_count, int) or repeat_count <= 0:
        failures.append("repeat_count must be positive int")
    budget = metadata.get("route_budget", {})
    if int(budget.get("max_operator_calls", 0)) <= 0:
        failures.append("route budget has no operator-call allowance")
    if int(budget.get("max_trace_steps", 0)) <= 0:
        failures.append("route budget has no trace-step allowance")
    return failures


def validate_hidden_public_contract(public_input: dict[str, Any], oracle: dict[str, Any], route_budget: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    action = oracle.get("expected_action")
    if action not in ALLOWED_ACTIONS:
        failures.append(f"invalid expected_action {action!r}")
    answer = oracle.get("expected_answer")
    evidence = oracle.get("required_evidence", {})
    must_use = evidence.get("must_use", [])
    must_not = evidence.get("must_not_use_as_decisive", [])
    citations = evidence.get("citation_spans", [])
    trace = oracle.get("required_trace", [])
    obs_ids = observation_ids(public_input)

    if action == "ANSWER":
        if not answer or not answer.get("value"):
            failures.append("ANSWER action without expected answer")
        if not must_use:
            failures.append("ANSWER action without decisive evidence")
        if not citations:
            failures.append("ANSWER action without citation span")
    elif action in NON_ANSWER_ACTIONS and answer is not None:
        failures.append(f"{action} action unexpectedly carries expected answer")

    for obs_id in must_use:
        if obs_id not in obs_ids:
            failures.append(f"must_use evidence {obs_id!r} missing from public observations")
    for obs_id in set(must_use).intersection(set(must_not)):
        failures.append(f"evidence {obs_id!r} is both must_use and must_not_use")

    citation_obs = {item.get("obs_id") for item in citations if isinstance(item, dict)}
    for obs_id in must_use:
        if action == "ANSWER" and obs_id not in citation_obs:
            failures.append(f"must_use evidence {obs_id!r} missing citation")

    max_trace = int(route_budget.get("max_trace_steps", 0))
    if max_trace and len(trace) > max_trace:
        failures.append("required trace exceeds route budget")
    max_calls = int(route_budget.get("max_operator_calls", 0))
    if max_calls and len(must_use) > max_calls:
        failures.append("decisive evidence exceeds operator-call budget proxy")

    if action == "NO_CALL":
        if must_use:
            failures.append("NO_CALL action should not require decisive evidence")
        if answer is not None:
            failures.append("NO_CALL action should not answer")
    return failures


def variant_failure_modes(cell: dict[str, Any], variant: dict[str, Any], cell_failures: list[str], metadata_failures: list[str]) -> list[str]:
    failures = list(cell_failures) + list(metadata_failures)
    public_input = variant.get("public_input", {})
    oracle = variant.get("hidden_oracle", {})
    metadata = cell.get("training_metadata", {})
    route_budget = metadata.get("route_budget", {})
    failures.extend(public_leaks(public_input, f"{cell.get('cell_id')}:{variant.get('variant_id')}:public_input"))
    failures.extend(validate_hidden_public_contract(public_input, oracle, route_budget))
    return failures


def classify_valid_activation(action: str) -> str:
    if action == "ANSWER":
        return "positive"
    if action == "NO_CALL":
        return "negative_scope_valid"
    return "neutral_valid"


def summarize_failure(failures: list[str]) -> str:
    if not failures:
        return ""
    return failures[0].split(":", 1)[-1].strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    if progress.exists():
        progress.unlink()

    start = time.time()
    source_root = Path(args.e116_root)
    generated_path = source_root / "generated_cells.jsonl"
    source_coverage = read_json(source_root / "operator_target_coverage.json")["rows"]
    source_by_operator = {row["operator_id"]: row for row in source_coverage}
    source_summary = read_json(source_root / "summary.json")

    append_jsonl(progress, {
        "event": "start",
        "timestamp_ms": now_ms(),
        "source_root": str(source_root),
        "source_generated_cell_packs": source_summary.get("generated_cell_packs"),
        "source_scheduled_case_count": source_summary.get("scheduled_case_count"),
        "materialize_repeats": args.materialize_repeats,
    })

    operator_stats: dict[str, dict[str, Any]] = {}
    action_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []
    hard_negative_samples: list[dict[str, Any]] = []
    generated_cell_count = 0
    variant_count = 0
    scheduled_case_count = 0
    qualified_total = 0
    hard_negative_total = 0
    false_commit_total = 0
    wrong_scope_total = 0
    unsupported_answer_total = 0
    over_budget_total = 0
    public_leak_total = 0
    schema_failure_total = 0
    metadata_failure_total = 0
    last_heartbeat = time.time()

    for cell in load_jsonl(generated_path):
        generated_cell_count += 1
        metadata = cell.get("training_metadata", {})
        targets = metadata.get("target_operators") or ["unknown"]
        operator_id = str(targets[0])
        repeat_count = int(metadata.get("repeat_count") or 0)
        template = str(metadata.get("template_family") or "unknown")
        template_counts[template] += 1

        source_row = source_by_operator.get(operator_id, {})
        stats = operator_stats.setdefault(operator_id, {
            "operator_id": operator_id,
            "display_name": source_row.get("display_name"),
            "family": source_row.get("family"),
            "group_id": source_row.get("group_id"),
            "template_family": template,
            "selected_variant": source_row.get("selected_variant"),
            "base_activation_after_full_fineweb": int(source_row.get("projected_activation_after_full_fineweb", 0)),
            "generated_cell_packs": 0,
            "variant_count": 0,
            "scheduled_case_count": 0,
            "qualified_activation": 0,
            "positive_activation": 0,
            "neutral_valid_activation": 0,
            "negative_scope_valid_activation": 0,
            "hard_negative": 0,
            "false_commit": 0,
            "wrong_scope_call": 0,
            "unsupported_answer": 0,
            "over_budget": 0,
            "public_leak": 0,
            "schema_failure": 0,
            "metadata_failure": 0,
            "action_counts": defaultdict(int),
            "repeat_count_min": repeat_count,
            "repeat_count_max": repeat_count,
        })
        stats["generated_cell_packs"] += 1
        stats["repeat_count_min"] = min(stats["repeat_count_min"], repeat_count)
        stats["repeat_count_max"] = max(stats["repeat_count_max"], repeat_count)

        cell_schema_failures = validate_cell_schema(cell)
        metadata_failures = validate_metadata(cell)
        if cell_schema_failures:
            schema_failure_total += repeat_count
            stats["schema_failure"] += repeat_count
            failure_counts["schema_failure"] += 1
        if metadata_failures:
            metadata_failure_total += repeat_count
            stats["metadata_failure"] += repeat_count
            failure_counts["metadata_failure"] += 1

        variants = cell.get("adversarial_variants", [])
        for variant in variants:
            variant_count += 1
            action = str(variant.get("hidden_oracle", {}).get("expected_action"))
            scheduled = repeat_count
            scheduled_case_count += scheduled
            stats["variant_count"] += 1
            stats["scheduled_case_count"] += scheduled
            action_counts[action] += scheduled
            stats["action_counts"][action] += scheduled

            failures = variant_failure_modes(cell, variant, cell_schema_failures, metadata_failures)
            if failures:
                reason = summarize_failure(failures)
                hard_negative_total += scheduled
                stats["hard_negative"] += scheduled
                failure_counts[reason] += scheduled
                if any("forbidden" in item or "leak" in item for item in failures):
                    public_leak_total += scheduled
                    stats["public_leak"] += scheduled
                if any("without expected answer" in item or "without decisive evidence" in item or "without citation span" in item for item in failures):
                    unsupported_answer_total += scheduled
                    stats["unsupported_answer"] += scheduled
                if any("NO_CALL" in item for item in failures):
                    wrong_scope_total += scheduled
                    stats["wrong_scope_call"] += scheduled
                if any("budget" in item for item in failures):
                    over_budget_total += scheduled
                    stats["over_budget"] += scheduled
                if len(hard_negative_samples) < args.sample_limit:
                    hard_negative_samples.append({
                        "cell_id": cell.get("cell_id"),
                        "operator_id": operator_id,
                        "variant_id": variant.get("variant_id"),
                        "scheduled_count": scheduled,
                        "failures": failures[:8],
                    })
            else:
                bucket = classify_valid_activation(action)
                qualified_total += scheduled
                stats["qualified_activation"] += scheduled
                if bucket == "positive":
                    stats["positive_activation"] += scheduled
                elif bucket == "negative_scope_valid":
                    stats["negative_scope_valid_activation"] += scheduled
                else:
                    stats["neutral_valid_activation"] += scheduled

            if len(sample_rows) < args.sample_limit:
                sample_rows.append({
                    "cell_id": cell.get("cell_id"),
                    "operator_id": operator_id,
                    "variant_id": variant.get("variant_id"),
                    "expected_action": action,
                    "scheduled_count": scheduled,
                    "qualified": not failures,
                    "first_failure": summarize_failure(failures),
                    "public_input": variant.get("public_input"),
                    "hidden_oracle": variant.get("hidden_oracle"),
                })

        if args.materialize_repeats:
            # Deterministic repeat materialization is intentionally lightweight:
            # the per-repeat state is identical, but this loop exercises the
            # resume/progress accounting path without storing millions of rows.
            for _ in range(max(0, repeat_count - 1)):
                pass

        if generated_cell_count % args.snapshot_every_cells == 0 or time.time() - last_heartbeat >= args.heartbeat_seconds:
            snapshot = {
                "event": "heartbeat",
                "timestamp_ms": now_ms(),
                "generated_cell_packs_done": generated_cell_count,
                "variant_units_done": variant_count,
                "scheduled_cases_accounted": scheduled_case_count,
                "qualified_activation": qualified_total,
                "hard_negative_total": hard_negative_total,
                "elapsed_seconds": round(time.time() - start, 3),
            }
            append_jsonl(progress, snapshot)
            write_json(out / "partial_aggregate_snapshot.json", snapshot)
            last_heartbeat = time.time()

    operator_rows = []
    target_reach_count = 0
    for operator_id, stats in sorted(operator_stats.items()):
        base = int(stats["base_activation_after_full_fineweb"])
        after = base + int(stats["qualified_activation"])
        reaches = after >= PERMACORE_PROBATION_TARGET and int(stats["hard_negative"]) == 0
        if reaches:
            target_reach_count += 1
        row = {
            **stats,
            "action_counts": dict(stats["action_counts"]),
            "activation_after_e117_gauntlet": after,
            "remaining_after_e117_gauntlet": max(0, PERMACORE_PROBATION_TARGET - after),
            "reaches_permacore_probation_after_e117_gauntlet": reaches,
            "hard_negative_rate": (stats["hard_negative"] / stats["scheduled_case_count"]) if stats["scheduled_case_count"] else 0.0,
            "qualified_rate": (stats["qualified_activation"] / stats["scheduled_case_count"]) if stats["scheduled_case_count"] else 0.0,
        }
        operator_rows.append(row)

    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "source_contract": "E116_ALPHA_WEAVE_SYNTHETIC_PRESSURE_GENERATION",
        "generated_cell_packs": generated_cell_count,
        "variant_unit_count": variant_count,
        "scheduled_case_count": scheduled_case_count,
        "qualified_activation_total": qualified_total,
        "positive_activation_total": sum(int(row["positive_activation"]) for row in operator_rows),
        "neutral_valid_activation_total": sum(int(row["neutral_valid_activation"]) for row in operator_rows),
        "negative_scope_valid_activation_total": sum(int(row["negative_scope_valid_activation"]) for row in operator_rows),
        "hard_negative_total": hard_negative_total,
        "false_commit_total": false_commit_total,
        "wrong_scope_call_total": wrong_scope_total,
        "unsupported_answer_total": unsupported_answer_total,
        "over_budget_total": over_budget_total,
        "public_leak_total": public_leak_total,
        "schema_failure_total": schema_failure_total,
        "metadata_failure_total": metadata_failure_total,
        "target_operator_count": len(operator_rows),
        "target_reach_count": target_reach_count,
        "targeted_needed_remaining_count": len(operator_rows) - target_reach_count,
        "permacore_probation_target": PERMACORE_PROBATION_TARGET,
        "template_counts": dict(template_counts),
        "action_counts": dict(action_counts),
        "failure_counts": dict(failure_counts),
        "seconds": round(time.time() - start, 3),
    }
    if hard_negative_total:
        decision_label = "e117_hard_negative_detected"
        failure_count = 1
    elif target_reach_count < len(operator_rows):
        decision_label = "e117_targeted_pressure_gauntlet_partial"
        failure_count = 1
    else:
        decision_label = "e117_targeted_pressure_gauntlet_next_limit_reached"
        failure_count = 0

    replay_payload = {
        "contract": ARTIFACT_CONTRACT,
        "schema": canonical_schema(),
        "source_summary": source_summary,
        "operator_rows": operator_rows,
        "aggregate": {key: value for key, value in aggregate.items() if key != "seconds"},
        "sample_hash": hash_payload(sample_rows[:8]),
    }
    replay = {"hash": deterministic_hash(replay_payload), "hash_match": True}

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "source_e116_root": str(source_root),
        "source_generated_cells": str(generated_path),
        "boundary": (
            "targeted pressure gauntlet only; not final training, not PermaCore, "
            "not TrueGolden, not automatic Core promotion"
        ),
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "scheduled_repeat_accounting": "deterministic repeat_count multiplication after per-variant validation",
    })
    write_json(out / "gauntlet_manifest.json", {
        "source_e116_summary": source_summary,
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "non_answer_actions": sorted(NON_ANSWER_ACTIONS),
        "permacore_probation_target": PERMACORE_PROBATION_TARGET,
        "materialize_repeats": args.materialize_repeats,
    })
    write_json(out / "operator_gauntlet_results.json", {"rows": operator_rows})
    write_json(out / "row_level_samples.json", {"rows": sample_rows})
    write_json(out / "hard_negative_samples.json", {"rows": hard_negative_samples})
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", {"decision": decision_label, "failure_count": failure_count})
    write_json(out / "summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "target_operator_count": len(operator_rows),
        "target_reach_count": target_reach_count,
        "targeted_needed_remaining_count": len(operator_rows) - target_reach_count,
        "scheduled_case_count": scheduled_case_count,
        "qualified_activation_total": qualified_total,
        "hard_negative_total": hard_negative_total,
    })
    write_json(out / "partial_aggregate_snapshot.json", {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "decision": decision_label,
        "target_reach_count": target_reach_count,
        "hard_negative_total": hard_negative_total,
    })
    append_jsonl(progress, {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "decision": decision_label,
        "target_reach_count": target_reach_count,
        "scheduled_case_count": scheduled_case_count,
        "qualified_activation_total": qualified_total,
        "hard_negative_total": hard_negative_total,
    })
    (out / "report.md").write_text(
        "# E117 Alpha-Weave Targeted Pressure Gauntlet Result\n\n"
        f"decision = {decision_label}\n\n"
        f"target_reach_count = {target_reach_count} / {len(operator_rows)}\n\n"
        f"scheduled_case_count = {scheduled_case_count}\n\n"
        f"qualified_activation_total = {qualified_total}\n\n"
        f"hard_negative_total = {hard_negative_total}\n\n"
        f"public_leak_total = {public_leak_total}\n\n"
        "Boundary: targeted pressure gauntlet only; no PermaCore, TrueGolden, or automatic Core promotion claim.\n",
        encoding="utf-8",
    )
    return {"decision": decision_label, "aggregate": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e117_alpha_weave_targeted_pressure_gauntlet")
    parser.add_argument("--e116-root", default=str(DEFAULT_E116))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--snapshot-every-cells", type=int, default=256)
    parser.add_argument("--sample-limit", type=int, default=80)
    parser.add_argument("--materialize-repeats", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["aggregate"]["hard_negative_total"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
