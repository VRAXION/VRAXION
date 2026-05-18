#!/usr/bin/env python3
"""Multi-seed eval-only orchestrator for STABLE_LOOP_PHASE_LOCK_074."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke")
DEFAULT_CHECKPOINT = Path(
    "target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json"
)
DEFAULT_UPSTREAM_072_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke")
DEFAULT_UPSTREAM_071B_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke")
DEFAULT_UPSTREAM_071_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke")
DEFAULT_UPSTREAM_070_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke")
REQUIRED_SEEDS = [2027, 2028, 2029]

POSITIVE_VERDICTS = [
    "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_POSITIVE",
    "FRESH_CHILD_RUNS_CONFIRMED",
    "CHILD_073_GATES_RECHECKED",
    "MULTI_SEED_MIN_GATE_PASSES",
    "SCENARIO_GATED_REPAIR_STABLE_ACROSS_SEEDS",
    "FRESH_GATED_ADVANTAGE_STABLE",
    "SCENARIO_SOURCE_ATTRIBUTION_AGGREGATED",
    "SHUFFLED_SCENARIO_CONTROL_FAILS_ALL_SEEDS",
    "CHECKPOINT_UNCHANGED_ALL_SEEDS",
    "NO_TRAINING_PERFORMED",
    "FRESH_EVAL_LEAKAGE_REJECTED_ALL_SEEDS",
    "BASELINE_COMPARISON_RECORDED",
    "FAILURE_CASE_REPORT_WRITTEN",
    "OPEN_ENDED_LIMITATION_RECORDED",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def quote_arg(arg: str) -> str:
    if any(ch.isspace() for ch in arg):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def command_string(cmd: list[str]) -> str:
    return " ".join(quote_arg(str(part)) for part in cmd)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def append_progress(out: Path, event: str, payload: dict[str, Any] | None = None) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "payload": payload or {}})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def write_report(out: Path, status: str, verdicts: list[str], seed_records: list[dict[str, Any]], message: str = "") -> None:
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_074_MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM Report",
        "",
        "074 is multi-seed eval-only confirmation of the 072 scenario-gated checkpoint.",
        "No training, checkpoint repair, open-ended assistant claim, full English LM claim, language grounding claim, production training claim, GA, public beta, hosted SaaS, or release readiness claim is made.",
        "",
        f"Status: `{status}`",
        "",
        "## Verdicts",
        "",
        "```text",
        *verdicts,
        "```",
        "",
    ]
    if message:
        lines.extend(["## Message", "", message, ""])
    if seed_records:
        lines.extend(
            [
                "## Seeds",
                "",
                "| seed | pass | supported | family_min | active | delta_ungated | shuffled | collapse |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in seed_records:
            lines.append(
                "| {seed} | `{seed_pass}` | `{supported:.3f}` | `{family:.3f}` | `{active:.3f}` | `{ungated:.3f}` | `{shuffled:.3f}` | `{collapse}` |".format(
                    seed=row["seed"],
                    seed_pass=row["seed_pass"],
                    supported=float(row.get("supported_accuracy", 0.0)),
                    family=float(row.get("family_min_accuracy", 0.0)),
                    active=float(row.get("active_scenario_selection_accuracy", 0.0)),
                    ungated=float(row.get("delta_vs_ungated_sidepacket_control", 0.0)),
                    shuffled=float(row.get("shuffled_scenario_label_control_accuracy", 1.0)),
                    collapse=row.get("collapse_detected"),
                )
            )
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(out: Path, status: str, verdicts: list[str], seed_records: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "schema_version": "multi_seed_scenario_gated_repair_confirm_summary_v1",
        "status": status,
        "multi_seed_eval_only": True,
        "train_step_count": 0,
        "open_ended_generation_supported": False,
        "free_form_answering_supported": False,
        "perplexity_supported": False,
        "full_English_LM_supported": False,
        "language_grounding_claimed": False,
        "production_training_claimed": False,
        "verdicts": verdicts,
        "seed_records": seed_records,
    }
    if extra:
        payload.update(extra)
    write_json(out / "summary.json", payload)
    write_report(out, status, verdicts, seed_records)


def fail(out: Path, verdicts: list[str], message: str, seed_records: list[dict[str, Any]] | None = None) -> int:
    final = ["MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_FAILS", *verdicts]
    records = seed_records or []
    append_progress(out, "failed", {"verdicts": final, "message": message})
    write_summary(out, "failed", final, records, {"message": message})
    write_report(out, "failed", final, records, message)
    return 1


def run_with_heartbeat(cmd: list[str], out: Path, event_prefix: str, heartbeat_sec: int, log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    append_progress(out, f"{event_prefix}_started", {"command": command_string(cmd), "log_path": str(log_path)})
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, shell=False)
        last = started
        while True:
            code = proc.poll()
            now = time.time()
            if now - last >= heartbeat_sec:
                append_progress(
                    out,
                    f"{event_prefix}_heartbeat",
                    {"pid": proc.pid, "elapsed_sec": round(now - started, 3), "log_path": str(log_path)},
                )
                last = now
            if code is not None:
                elapsed = time.time() - started
                append_progress(
                    out,
                    f"{event_prefix}_completed",
                    {"exit_code": code, "elapsed_sec": round(elapsed, 3), "log_path": str(log_path)},
                )
                return code, elapsed
            time.sleep(min(2, max(1, heartbeat_sec // 5)))


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if seeds != REQUIRED_SEEDS:
        raise ValueError("074 requires exact seeds 2027,2028,2029; no mean-only or best-seed shortcut")
    return seeds


def validate_upstreams(args: argparse.Namespace) -> list[str]:
    required = [
        args.checkpoint,
        args.upstream_072_root / "summary.json",
        args.upstream_072_root / "checkpoint_manifest.json",
        args.upstream_072_root / "arm_comparison.json",
        args.upstream_072_root / "targeted_dataset_manifest.json",
        args.upstream_071b_root / "summary.json",
        args.upstream_071_root / "summary.json",
        args.upstream_070_root / "summary.json",
    ]
    return [str(path) for path in required if not path.exists()]


def child_command(args: argparse.Namespace, seed: int, child_out: Path) -> list[str]:
    return [
        "cargo",
        "run",
        "-p",
        "instnct-core",
        "--example",
        "phase_lane_scenario_gated_repair_fresh_confirm",
        "--",
        "--out",
        str(child_out),
        "--checkpoint",
        str(args.checkpoint),
        "--upstream-072-root",
        str(args.upstream_072_root),
        "--upstream-071b-root",
        str(args.upstream_071b_root),
        "--upstream-071-root",
        str(args.upstream_071_root),
        "--upstream-070-root",
        str(args.upstream_070_root),
        "--seed",
        str(seed),
        "--heartbeat-sec",
        str(args.heartbeat_sec),
    ]


def collect_failures(child_out: Path, seed: int) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(child_out / "failure_case_samples.jsonl"):
        row = {"seed": seed, **row}
        rows.append(row)
    return rows


def recheck_child(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    verdicts = summary.get("verdicts", [])
    if "SCENARIO_GATED_REPAIR_FRESH_CONFIRM_POSITIVE" not in verdicts:
        failures.append("missing_positive_verdict")
    checks = {
        "train_step_count": summary.get("train_step_count") == 0,
        "checkpoint_hash_unchanged": summary.get("checkpoint_hash_unchanged") is True,
        "prediction_oracle_used": summary.get("prediction_oracle_used") is False,
        "baseline_eval_mismatch": summary.get("baseline_eval_mismatch") is False,
        "collapse_detected": summary.get("collapse_detected") is False,
    }
    for key, ok in checks.items():
        if not ok:
            failures.append(key)
    overlap_keys = [
        "overlap_with_070_eval_count",
        "overlap_with_071_eval_count",
        "overlap_with_071b_failure_digest_count",
        "overlap_with_072_train_count",
        "overlap_with_072_eval_count",
    ]
    for key in overlap_keys:
        if summary.get(key) != 0:
            failures.append(key)
    metric_gates = {
        "fresh_active_scenario_binding_accuracy": float(summary.get("fresh_active_scenario_binding_accuracy", 0.0)) >= 0.90,
        "fresh_counterfactual_scenario_switch_accuracy": float(summary.get("fresh_counterfactual_scenario_switch_accuracy", 0.0)) >= 0.85,
        "fresh_distractor_scenario_rejection_accuracy": float(summary.get("fresh_distractor_scenario_rejection_accuracy", 0.0)) >= 0.90,
        "fresh_old_scenario_suppression_accuracy": float(summary.get("fresh_old_scenario_suppression_accuracy", 0.0)) >= 0.90,
        "fresh_inactive_pocket_suppression_accuracy": float(summary.get("fresh_inactive_pocket_suppression_accuracy", 0.0)) >= 0.85,
        "fresh_stale_pocket_suppression_accuracy": float(summary.get("fresh_stale_pocket_suppression_accuracy", 0.0)) >= 0.85,
        "fresh_answer_only_scenario_binding_accuracy": float(summary.get("fresh_answer_only_scenario_binding_accuracy", 0.0)) >= 0.85,
        "family_min_accuracy": float(summary.get("family_min_accuracy", 0.0)) >= 0.85,
        "supported_accuracy": float(summary.get("supported_accuracy", 0.0)) >= 0.88,
        "active_scenario_selection_accuracy": float(summary.get("active_scenario_selection_accuracy", 0.0)) >= 0.95,
        "distractor_scenario_selection_rate": float(summary.get("distractor_scenario_selection_rate", 1.0)) <= 0.05,
        "old_scenario_selection_rate": float(summary.get("old_scenario_selection_rate", 1.0)) <= 0.05,
        "delta_vs_no_route_control": float(summary.get("delta_vs_no_route_control", 0.0)) > 0.10,
        "delta_vs_ungated_sidepacket_control": float(summary.get("delta_vs_ungated_sidepacket_control", 0.0)) > 0.03,
        "delta_vs_copy_first_match": float(summary.get("delta_vs_copy_first_match", 0.0)) > 0.10,
        "shuffled_scenario_label_control_accuracy": float(summary.get("shuffled_scenario_label_control_accuracy", 1.0)) < 0.70,
    }
    for key, ok in metric_gates.items():
        if not ok:
            failures.append(key)
    return not failures, failures


def seed_record(seed: int, child_out: Path, command: list[str], exit_code: int, elapsed: float, started: float, completed: float) -> dict[str, Any]:
    summary_path = child_out / "summary.json"
    report_path = child_out / "report.md"
    summary = read_json(summary_path) if summary_path.exists() else {}
    child_recheck_pass, child_recheck_failures = recheck_child(summary) if summary else (False, ["summary_missing"])
    return {
        "seed": seed,
        "child_run_started": True,
        "child_run_completed": exit_code == 0,
        "child_exit_code": exit_code,
        "child_elapsed_sec": round(elapsed, 3),
        "child_command": command_string(command),
        "child_summary_path": str(summary_path),
        "child_report_path": str(report_path),
        "child_summary_newer_than_074_start": summary_path.exists() and file_mtime(summary_path) >= started,
        "child_report_newer_than_074_start": report_path.exists() and file_mtime(report_path) >= started,
        "child_completed_after_started": completed >= started,
        "child_recheck_pass": child_recheck_pass,
        "child_recheck_failures": child_recheck_failures,
        "seed_pass": exit_code == 0
        and summary_path.exists()
        and report_path.exists()
        and file_mtime(summary_path) >= started
        and file_mtime(report_path) >= started
        and child_recheck_pass,
        **{key: summary.get(key) for key in [
            "supported_accuracy",
            "family_min_accuracy",
            "fresh_active_scenario_binding_accuracy",
            "fresh_counterfactual_scenario_switch_accuracy",
            "fresh_distractor_scenario_rejection_accuracy",
            "fresh_old_scenario_suppression_accuracy",
            "fresh_inactive_pocket_suppression_accuracy",
            "fresh_stale_pocket_suppression_accuracy",
            "fresh_answer_only_scenario_binding_accuracy",
            "active_scenario_selection_accuracy",
            "distractor_scenario_selection_rate",
            "old_scenario_selection_rate",
            "inactive_pocket_selection_rate",
            "stale_pocket_selection_rate",
            "first_ledger_bias_rate",
            "side_note_leak_rate",
            "delta_vs_no_route_control",
            "delta_vs_ungated_sidepacket_control",
            "delta_vs_copy_first_match",
            "shuffled_scenario_label_control_accuracy",
            "collapse_detected",
            "checkpoint_hash_unchanged",
            "train_step_count",
            "prediction_oracle_used",
            "baseline_eval_mismatch",
            "overlap_with_070_eval_count",
            "overlap_with_071_eval_count",
            "overlap_with_071b_failure_digest_count",
            "overlap_with_072_train_count",
            "overlap_with_072_eval_count",
        ]},
    }


def aggregate_metrics(seed_records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "supported_accuracy",
        "family_min_accuracy",
        "active_scenario_selection_accuracy",
        "delta_vs_no_route_control",
        "delta_vs_ungated_sidepacket_control",
        "delta_vs_copy_first_match",
    ]
    out: dict[str, Any] = {"all_seed_pass": all(row.get("seed_pass") for row in seed_records)}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in seed_records]
        out[f"min_{key}"] = min(values) if values else 0.0
        out[f"max_{key}"] = max(values) if values else 0.0
        out[f"mean_{key}"] = mean(values)
        out[f"stddev_{key}"] = stddev(values)
    out["seed_count"] = len(seed_records)
    return out


def source_attribution_aggregate(seed_records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "active_scenario_selection_accuracy",
        "distractor_scenario_selection_rate",
        "old_scenario_selection_rate",
        "inactive_pocket_selection_rate",
        "stale_pocket_selection_rate",
        "first_ledger_bias_rate",
        "side_note_leak_rate",
    ]
    payload: dict[str, Any] = {"per_seed": seed_records}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in seed_records]
        payload[key] = {
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "mean": mean(values),
            "stddev": stddev(values),
        }
    return payload


def final_verdicts(seed_records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    if not all(row.get("child_summary_newer_than_074_start") and row.get("child_report_newer_than_074_start") for row in seed_records):
        failures.append("STALE_CHILD_ARTIFACT_USED")
    if not all(row.get("child_recheck_pass") for row in seed_records):
        failures.append("CHILD_073_GATE_RECHECK_FAILS")
    if not all(row.get("seed_pass") for row in seed_records):
        failures.append("MULTI_SEED_SCENARIO_INSTABILITY_DETECTED")
    if not all(float(row.get("delta_vs_no_route_control", 0.0)) > 0.10 and float(row.get("delta_vs_ungated_sidepacket_control", 0.0)) > 0.03 and float(row.get("delta_vs_copy_first_match", 0.0)) > 0.10 for row in seed_records):
        failures.append("GATED_ADVANTAGE_REGRESSION_DETECTED")
    if not all(float(row.get("shuffled_scenario_label_control_accuracy", 1.0)) < 0.70 for row in seed_records):
        failures.append("SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS")
    if not all(row.get("checkpoint_hash_unchanged") is True for row in seed_records):
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    if not all(row.get("train_step_count") == 0 for row in seed_records):
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    if not all(row.get("prediction_oracle_used") is False for row in seed_records):
        failures.append("ORACLE_SHORTCUT_DETECTED")
    if any(row.get(key, 0) != 0 for row in seed_records for key in [
        "overlap_with_070_eval_count",
        "overlap_with_071_eval_count",
        "overlap_with_071b_failure_digest_count",
        "overlap_with_072_train_count",
        "overlap_with_072_eval_count",
    ]):
        failures.append("BENCHMARK_LEAKAGE_DETECTED")
    if not all(row.get("baseline_eval_mismatch") is False for row in seed_records):
        failures.append("BASELINE_EVAL_MISMATCH")
    if not all(row.get("collapse_detected") is False for row in seed_records):
        failures.append("STATIC_OUTPUT_COLLAPSE_DETECTED")
    if failures:
        return ["MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_FAILS", *failures]
    return POSITIVE_VERDICTS.copy()


def write_aggregate_files(out: Path, seed_records: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_metrics(seed_records)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(
        out / "multi_seed_stability.json",
        {
            "schema_version": "multi_seed_scenario_stability_v1",
            "all_seed_pass": aggregate["all_seed_pass"],
            "min_supported_accuracy": aggregate["min_supported_accuracy"],
            "min_family_min_accuracy": aggregate["min_family_min_accuracy"],
            "min_active_scenario_selection_accuracy": aggregate["min_active_scenario_selection_accuracy"],
            "stddev_supported_accuracy": aggregate["stddev_supported_accuracy"],
            "stddev_family_min_accuracy": aggregate["stddev_family_min_accuracy"],
            "stddev_delta_vs_ungated_sidepacket_control": aggregate["stddev_delta_vs_ungated_sidepacket_control"],
        },
    )
    write_json(
        out / "baseline_knockout_aggregate.json",
        {
            "schema_version": "multi_seed_baseline_knockout_aggregate_v1",
            "min_delta_vs_no_route_control": aggregate["min_delta_vs_no_route_control"],
            "min_delta_vs_ungated_sidepacket_control": aggregate["min_delta_vs_ungated_sidepacket_control"],
            "min_delta_vs_copy_first_match": aggregate["min_delta_vs_copy_first_match"],
            "per_seed": [
                {
                    "seed": row["seed"],
                    "delta_vs_no_route_control": row.get("delta_vs_no_route_control"),
                    "delta_vs_ungated_sidepacket_control": row.get("delta_vs_ungated_sidepacket_control"),
                    "delta_vs_copy_first_match": row.get("delta_vs_copy_first_match"),
                    "shuffled_scenario_label_control_accuracy": row.get("shuffled_scenario_label_control_accuracy"),
                }
                for row in seed_records
            ],
        },
    )
    write_json(out / "scenario_source_attribution_aggregate.json", source_attribution_aggregate(seed_records))
    write_json(
        out / "retention_aggregate.json",
        {
            "schema_version": "multi_seed_retention_aggregate_v1",
            "retention_inferred_from_child_verdict": "RETENTION_CONFIRM_PASSES",
            "all_seed_retention_pass": all(row.get("seed_pass") for row in seed_records),
            "per_seed": [{"seed": row["seed"], "seed_pass": row["seed_pass"]} for row in seed_records],
        },
    )
    write_jsonl = out / "failure_case_samples.jsonl"
    with write_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for row in failure_rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--upstream-072-root", type=Path, default=DEFAULT_UPSTREAM_072_ROOT)
    parser.add_argument("--upstream-071b-root", type=Path, default=DEFAULT_UPSTREAM_071B_ROOT)
    parser.add_argument("--upstream-071-root", type=Path, default=DEFAULT_UPSTREAM_071_ROOT)
    parser.add_argument("--upstream-070-root", type=Path, default=DEFAULT_UPSTREAM_070_ROOT)
    parser.add_argument("--seeds", default="2027,2028,2029")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    append_progress(out, "start", {"seeds": args.seeds})
    write_summary(out, "running", ["MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_RUNNING"], [])
    write_json(
        out / "queue.json",
        {
            "schema_version": "multi_seed_scenario_gated_repair_confirm_queue_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_074_MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM",
            "steps": ["validate_upstreams", "run_child_073_per_seed", "recheck_child_gates", "aggregate", "done"],
        },
    )
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as err:
        return fail(out, ["MULTI_SEED_SCENARIO_INSTABILITY_DETECTED"], str(err))

    missing = validate_upstreams(args)
    if missing:
        return fail(out, ["UPSTREAM_072_ARTIFACT_MISSING"], ", ".join(missing))

    write_json(
        out / "multi_seed_config.json",
        {
            "schema_version": "multi_seed_scenario_gated_repair_confirm_config_v1",
            "multi_seed_eval_only": True,
            "train_step_count": 0,
            "open_ended_generation_supported": False,
            "free_form_answering_supported": False,
            "perplexity_supported": False,
            "full_English_LM_supported": False,
            "language_grounding_claimed": False,
            "production_training_claimed": False,
            "checkpoint": str(args.checkpoint),
            "seeds": seeds,
            "heartbeat_sec": args.heartbeat_sec,
        },
    )
    write_json(
        out / "upstream_072_manifest.json",
        {
            "schema_version": "multi_seed_upstream_072_manifest_v1",
            "checkpoint": str(args.checkpoint),
            "upstream_072_root": str(args.upstream_072_root),
            "upstream_071b_root": str(args.upstream_071b_root),
            "upstream_071_root": str(args.upstream_071_root),
            "upstream_070_root": str(args.upstream_070_root),
        },
    )

    seed_records: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for seed in seeds:
        child_out = out / f"seed_{seed}"
        if child_out.exists():
            shutil.rmtree(child_out)
        cmd = child_command(args, seed, child_out)
        child_started = time.time()
        code, elapsed = run_with_heartbeat(cmd, out, f"seed_{seed}", args.heartbeat_sec, out / "logs" / f"seed_{seed}.log")
        child_completed = time.time()
        record = seed_record(seed, child_out, cmd, code, elapsed, child_started, child_completed)
        seed_records.append(record)
        failure_rows.extend(collect_failures(child_out, seed))
        append_jsonl(out / "seed_metrics.jsonl", record)
        write_json(out / "child_run_manifest.json", {"schema_version": "child_run_manifest_v1", "children": seed_records})
        write_aggregate_files(out, seed_records, failure_rows)
        write_summary(out, "running", ["MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_RUNNING"], seed_records)
        append_progress(out, "seed_completed", {"seed": seed, "seed_pass": record.get("seed_pass")})

    aggregate = write_aggregate_files(out, seed_records, failure_rows)
    verdicts = final_verdicts(seed_records)
    status = "passed" if verdicts == POSITIVE_VERDICTS else "failed"
    extra = {
        **aggregate,
        "child_run_manifest": seed_records,
        "total_elapsed_sec": round(time.time() - start, 3),
        "multi_seed_eval_only": True,
        "train_step_count": 0,
        "open_ended_generation_supported": False,
        "free_form_answering_supported": False,
        "perplexity_supported": False,
        "full_English_LM_supported": False,
        "language_grounding_claimed": False,
        "production_training_claimed": False,
    }
    write_summary(out, status, verdicts, seed_records, extra)
    append_progress(out, "done", {"status": status, "verdicts": verdicts})
    print(json.dumps(read_json(out / "summary.json"), separators=(",", ":")))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
