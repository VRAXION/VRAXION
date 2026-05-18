#!/usr/bin/env python3
"""Multi-seed eval-only orchestrator for STABLE_LOOP_PHASE_LOCK_082."""

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
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke")
DEFAULT_UPSTREAM_080_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke")
DEFAULT_UPSTREAM_079B_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke")
DEFAULT_UPSTREAM_079_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke")
DEFAULT_UPSTREAM_078_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke")
DEFAULT_UPSTREAM_074_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke")
REQUIRED_SEEDS = [2027, 2028, 2029]

POSITIVE_VERDICTS = [
    "CHAT_DIVERSITY_MULTI_SEED_CONFIRM_POSITIVE",
    "FRESH_CHILD_RUNS_CONFIRMED",
    "CHILD_081_GATES_RECHECKED",
    "MULTI_SEED_MIN_GATE_PASSES",
    "CHAT_DIVERSITY_STABLE_ACROSS_SEEDS",
    "TEMPLATE_COPY_REJECTED_ALL_SEEDS",
    "SKELETON_REUSE_REJECTED_ALL_SEEDS",
    "VOCAB_ENTROPY_PASSES_ALL_SEEDS",
    "CONTEXT_SLOT_BINDING_PASSES_ALL_SEEDS",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES_ALL_SEEDS",
    "CHECKPOINT_UNCHANGED_ALL_SEEDS",
    "NO_TRAINING_PERFORMED",
    "FAILURE_CASE_REPORT_WRITTEN",
    "PRODUCTION_CHAT_NOT_CLAIMED",
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
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


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


def metric(summary: dict[str, Any], group: str, key: str, default: float = 0.0) -> float:
    return float(summary.get(group, {}).get(key, default))


def write_report(out: Path, status: str, verdicts: list[str], seed_records: list[dict[str, Any]], message: str = "") -> None:
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM Report",
        "",
        "082 is bounded multi-seed chat diversity confirmation only.",
        "It makes no GPT-like assistant readiness, full English LM, language grounding, production chat, safety alignment, public beta, GA, or hosted SaaS claim.",
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
                "| seed | pass | novel | template_copy | skeleton_reuse | slot | retention | checkpoint |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in seed_records:
            lines.append(
                "| {seed} | `{seed_pass}` | `{novel:.3f}` | `{template:.3f}` | `{skeleton:.3f}` | `{slot:.3f}` | `{retention:.3f}` | `{checkpoint}` |".format(
                    seed=row["seed"],
                    seed_pass=row.get("seed_pass"),
                    novel=float(row.get("novel_response_rate", 0.0)),
                    template=float(row.get("template_copy_rate", 1.0)),
                    skeleton=float(row.get("response_skeleton_reuse_rate", 1.0)),
                    slot=float(row.get("slot_binding_accuracy", 0.0)),
                    retention=float(row.get("finite_label_retention_accuracy", 0.0)),
                    checkpoint=row.get("checkpoint_hash_unchanged"),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Boundaries",
            "",
            "bounded multi-seed chat diversity confirmation only",
            "not GPT-like assistant readiness",
            "not full English LM",
            "not language grounding",
            "not production chat",
            "not safety alignment",
            "not public beta / GA / hosted SaaS",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    out: Path,
    status: str,
    verdicts: list[str],
    seed_records: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "chat_diversity_multi_seed_confirm_summary_v1",
        "status": status,
        "multi_seed_eval_only": True,
        "train_step_count": 0,
        "open_ended_generation_supported": False,
        "full_English_LM_supported": False,
        "language_grounding_claimed": False,
        "production_chat_claimed": False,
        "safety_alignment_claimed": False,
        "verdicts": verdicts,
        "seed_records": seed_records,
    }
    if extra:
        payload.update(extra)
    write_json(out / "summary.json", payload)
    write_report(out, status, verdicts, seed_records)


def fail(out: Path, verdicts: list[str], message: str, seed_records: list[dict[str, Any]] | None = None) -> int:
    final = ["CHAT_DIVERSITY_MULTI_SEED_CONFIRM_FAILS", *verdicts]
    records = seed_records or []
    append_progress(out, "failed", {"verdicts": final, "message": message})
    write_summary(out, "failed", final, records, {"message": message})
    write_report(out, "failed", final, records, message)
    ensure_failure_file(out, [])
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
        raise ValueError("082 requires exact seeds 2027,2028,2029; no mean-only or best-seed shortcut")
    return seeds


def checkpoint_path(args: argparse.Namespace) -> Path:
    return args.upstream_080_root / "checkpoints" / "chat_composition_diversity_repair" / "model_checkpoint.json"


def validate_upstreams(args: argparse.Namespace) -> list[str]:
    required = [
        checkpoint_path(args),
        args.upstream_080_root / "summary.json",
        args.upstream_080_root / "checkpoint_manifest.json",
        args.upstream_080_root / "generation_samples.jsonl",
        args.upstream_079b_root / "summary.json",
        args.upstream_079_root / "summary.json",
        args.upstream_078_root / "summary.json",
        args.upstream_074_root / "summary.json",
    ]
    return [str(path) for path in required if not path.exists()]


def child_command(args: argparse.Namespace, seed: int, child_out: Path) -> list[str]:
    return [
        "cargo",
        "run",
        "-p",
        "instnct-core",
        "--example",
        "phase_lane_chat_diversity_fresh_confirm",
        "--",
        "--out",
        str(child_out),
        "--upstream-080-root",
        str(args.upstream_080_root),
        "--upstream-079b-root",
        str(args.upstream_079b_root),
        "--upstream-079-root",
        str(args.upstream_079_root),
        "--upstream-078-root",
        str(args.upstream_078_root),
        "--upstream-074-root",
        str(args.upstream_074_root),
        "--seed",
        str(seed),
        "--heartbeat-sec",
        str(args.heartbeat_sec),
    ]


def recheck_child(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if "CHAT_DIVERSITY_FRESH_CONFIRM_POSITIVE" not in summary.get("verdicts", []):
        failures.append("missing_positive_verdict")
    scalar_checks = {
        "train_step_count": summary.get("train_step_count") == 0,
        "checkpoint_hash_unchanged": summary.get("checkpoint_hash_unchanged") is True,
        "prediction_oracle_used": summary.get("prediction_oracle_used") is False,
        "llm_judge_used": summary.get("llm_judge_used") is False,
        "response_table_used_for_main_prediction": summary.get("response_table_used_for_main_prediction") is False,
    }
    for key, ok in scalar_checks.items():
        if not ok:
            failures.append(key)
    novelty = summary.get("novelty_metrics", {})
    skeleton = summary.get("skeleton_diversity_metrics", {})
    vocab = summary.get("vocabulary_entropy_metrics", {})
    context = summary.get("context_slot_metrics", {})
    retention = summary.get("finite_label_retention_metrics", {})
    collapse = summary.get("collapse_metrics", {})
    leakage = summary.get("prompt_leakage_metrics", {})
    gates = {
        "template_copy_rate": float(novelty.get("template_copy_rate", 1.0)) <= 0.25,
        "response_table_copy_rate": float(novelty.get("response_table_copy_rate", 1.0)) <= 0.20,
        "semantic_template_overlap_rate": float(novelty.get("semantic_template_overlap_rate", 1.0)) <= 0.50,
        "slot_only_skeleton_reuse_rate": float(novelty.get("slot_only_skeleton_reuse_rate", 1.0)) <= 0.25,
        "response_skeleton_reuse_rate": float(skeleton.get("response_skeleton_reuse_rate", 1.0)) <= 0.50,
        "top_skeleton_rate": float(skeleton.get("top_skeleton_rate", 1.0)) <= 0.35,
        "response_skeleton_diversity": float(skeleton.get("response_skeleton_diversity", 0.0)) >= 0.50,
        "generated_to_train_vocab_ratio": float(vocab.get("generated_to_train_vocab_ratio", 0.0)) >= 0.35,
        "unique_bigram_count": float(vocab.get("unique_bigram_count", 0.0)) >= 30,
        "unique_trigram_count": float(vocab.get("unique_trigram_count", 0.0)) >= 30,
        "token_entropy": float(vocab.get("token_entropy", 0.0)) > 2.0,
        "response_entropy": float(vocab.get("response_entropy", 0.0)) > 2.0,
        "slot_binding_accuracy": float(context.get("slot_binding_accuracy", 0.0)) >= 0.75,
        "finite_label_retention_accuracy": float(retention.get("finite_label_retention_accuracy", 0.0)) >= 0.90,
        "empty_output_rate": float(collapse.get("empty_output_rate", 1.0)) <= 0.02,
        "space_output_rate": float(collapse.get("space_output_rate", 1.0)) <= 0.02,
        "static_response_rate": float(collapse.get("static_response_rate", 1.0)) <= 0.15,
        "repetition_rate": float(collapse.get("repetition_rate", 1.0)) <= 0.20,
        "copy_prompt_rate": float(collapse.get("copy_prompt_rate", 1.0)) <= 0.15,
    }
    for key, ok in gates.items():
        if not ok:
            failures.append(key)
    overlap_keys = [
        "overlap_with_080_train_prompt_count",
        "overlap_with_080_eval_prompt_count",
        "overlap_with_079_prompt_count",
        "overlap_with_078_prompt_count",
        "overlap_with_076_prompt_count",
    ]
    for key in overlap_keys:
        if leakage.get(key) != 0:
            failures.append(key)
    if leakage.get("near_duplicate_prompt_count") != 0:
        failures.append("near_duplicate_prompt_count")
    return not failures, failures


def seed_record(seed: int, child_out: Path, command: list[str], exit_code: int, elapsed: float, started: float, completed: float) -> dict[str, Any]:
    summary_path = child_out / "summary.json"
    report_path = child_out / "report.md"
    summary = read_json(summary_path) if summary_path.exists() else {}
    child_recheck_pass, child_recheck_failures = recheck_child(summary) if summary else (False, ["summary_missing"])
    novelty = summary.get("novelty_metrics", {})
    skeleton = summary.get("skeleton_diversity_metrics", {})
    vocab = summary.get("vocabulary_entropy_metrics", {})
    context = summary.get("context_slot_metrics", {})
    retention = summary.get("finite_label_retention_metrics", {})
    collapse = summary.get("collapse_metrics", {})
    leakage = summary.get("prompt_leakage_metrics", {})
    return {
        "seed": seed,
        "child_run_started": True,
        "child_run_completed": exit_code == 0,
        "child_exit_code": exit_code,
        "child_elapsed_sec": round(elapsed, 3),
        "child_command": command_string(command),
        "child_summary_path": str(summary_path),
        "child_report_path": str(report_path),
        "child_summary_newer_than_082_start": summary_path.exists() and file_mtime(summary_path) >= started,
        "child_report_newer_than_082_start": report_path.exists() and file_mtime(report_path) >= started,
        "child_completed_after_started": completed >= started,
        "child_recheck_pass": child_recheck_pass,
        "child_recheck_failures": child_recheck_failures,
        "seed_pass": exit_code == 0
        and summary_path.exists()
        and report_path.exists()
        and file_mtime(summary_path) >= started
        and file_mtime(report_path) >= started
        and child_recheck_pass,
        "checkpoint_hash_before": summary.get("checkpoint_hash_before"),
        "checkpoint_hash_after": summary.get("checkpoint_hash_after"),
        "checkpoint_hash_unchanged": summary.get("checkpoint_hash_unchanged"),
        "train_step_count": summary.get("train_step_count"),
        "prediction_oracle_used": summary.get("prediction_oracle_used"),
        "llm_judge_used": summary.get("llm_judge_used"),
        "response_table_used_for_main_prediction": summary.get("response_table_used_for_main_prediction"),
        "novel_response_rate": novelty.get("novel_response_rate"),
        "template_copy_rate": novelty.get("template_copy_rate"),
        "response_table_copy_rate": novelty.get("response_table_copy_rate"),
        "semantic_template_overlap_rate": novelty.get("semantic_template_overlap_rate"),
        "slot_only_skeleton_reuse_rate": novelty.get("slot_only_skeleton_reuse_rate"),
        "response_skeleton_reuse_rate": skeleton.get("response_skeleton_reuse_rate"),
        "top_skeleton_rate": skeleton.get("top_skeleton_rate"),
        "response_skeleton_diversity": skeleton.get("response_skeleton_diversity"),
        "generated_to_train_vocab_ratio": vocab.get("generated_to_train_vocab_ratio"),
        "unique_bigram_count": vocab.get("unique_bigram_count"),
        "unique_trigram_count": vocab.get("unique_trigram_count"),
        "token_entropy": vocab.get("token_entropy"),
        "response_entropy": vocab.get("response_entropy"),
        "slot_binding_accuracy": context.get("slot_binding_accuracy"),
        "finite_label_retention_accuracy": retention.get("finite_label_retention_accuracy"),
        "empty_output_rate": collapse.get("empty_output_rate"),
        "space_output_rate": collapse.get("space_output_rate"),
        "static_response_rate": collapse.get("static_response_rate"),
        "repetition_rate": collapse.get("repetition_rate"),
        "copy_prompt_rate": collapse.get("copy_prompt_rate"),
        "near_duplicate_prompt_count": leakage.get("near_duplicate_prompt_count"),
        **{key: leakage.get(key) for key in [
            "overlap_with_080_train_prompt_count",
            "overlap_with_080_eval_prompt_count",
            "overlap_with_079_prompt_count",
            "overlap_with_078_prompt_count",
            "overlap_with_076_prompt_count",
        ]},
    }


def aggregate_numeric(seed_records: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in keys:
        values = [float(row.get(key, 0.0) or 0.0) for row in seed_records]
        payload[f"min_{key}"] = min(values) if values else 0.0
        payload[f"max_{key}"] = max(values) if values else 0.0
        payload[f"mean_{key}"] = mean(values)
        payload[f"stddev_{key}"] = stddev(values)
    return payload


def aggregate_metrics(seed_records: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "all_seed_pass": all(row.get("seed_pass") for row in seed_records),
        "seed_count": len(seed_records),
    }
    payload.update(
        aggregate_numeric(
            seed_records,
            [
                "novel_response_rate",
                "template_copy_rate",
                "response_skeleton_diversity",
                "response_skeleton_reuse_rate",
                "slot_binding_accuracy",
                "finite_label_retention_accuracy",
                "response_entropy",
                "generated_to_train_vocab_ratio",
            ],
        )
    )
    return payload


def record_float(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    return float(default if value is None else value)


def final_verdicts(seed_records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    if not all(row.get("child_summary_newer_than_082_start") and row.get("child_report_newer_than_082_start") for row in seed_records):
        failures.append("STALE_CHILD_ARTIFACT_USED")
    if not all(row.get("child_recheck_pass") for row in seed_records):
        failures.append("CHILD_081_GATE_RECHECK_FAILS")
    if not all(row.get("seed_pass") for row in seed_records):
        failures.append("MULTI_SEED_CHAT_DIVERSITY_INSTABILITY_DETECTED")
    if not all(row.get("checkpoint_hash_unchanged") is True for row in seed_records):
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    if any((row.get("near_duplicate_prompt_count") or 0) != 0 for row in seed_records) or any(
        (row.get(key) or 0) != 0
        for row in seed_records
        for key in [
            "overlap_with_080_train_prompt_count",
            "overlap_with_080_eval_prompt_count",
            "overlap_with_079_prompt_count",
            "overlap_with_078_prompt_count",
            "overlap_with_076_prompt_count",
        ]
    ):
        failures.append("FRESH_PROMPT_LEAKAGE_DETECTED")
    if not all(record_float(row, "template_copy_rate", 1.0) <= 0.25 for row in seed_records):
        failures.append("TEMPLATE_COPY_REGRESSION_DETECTED")
    if not all(record_float(row, "response_skeleton_reuse_rate", 1.0) <= 0.50 for row in seed_records):
        failures.append("SKELETON_REUSE_REGRESSION_DETECTED")
    if not all(record_float(row, "generated_to_train_vocab_ratio", 0.0) >= 0.35 for row in seed_records):
        failures.append("VOCAB_DIVERSITY_REGRESSION_DETECTED")
    if not all(record_float(row, "slot_binding_accuracy", 0.0) >= 0.75 for row in seed_records):
        failures.append("CONTEXT_SLOT_BINDING_REGRESSION_DETECTED")
    if not all(record_float(row, "finite_label_retention_accuracy", 0.0) >= 0.90 for row in seed_records):
        failures.append("FINITE_LABEL_RETENTION_REGRESSION_DETECTED")
    if not all(
        record_float(row, "empty_output_rate", 1.0) <= 0.02
        and record_float(row, "space_output_rate", 1.0) <= 0.02
        and record_float(row, "static_response_rate", 1.0) <= 0.15
        and record_float(row, "repetition_rate", 1.0) <= 0.20
        for row in seed_records
    ):
        failures.append("STATIC_RESPONSE_COLLAPSE_DETECTED")
    if failures:
        return ["CHAT_DIVERSITY_MULTI_SEED_CONFIRM_FAILS", *failures]
    return POSITIVE_VERDICTS.copy()


def collect_failure_rows(child_out: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(child_out / "generation_samples.jsonl"):
        if row.get("pass_fail") == "fail":
            rows.append(
                {
                    "seed": seed,
                    "eval_family": row.get("eval_family"),
                    "prompt": row.get("prompt"),
                    "model_output": row.get("model_output"),
                    "expected_behavior": row.get("expected_behavior"),
                    "pass_fail": row.get("pass_fail"),
                    "novelty_flag": row.get("novelty_flag"),
                    "template_copy_flag": row.get("template_copy_flag"),
                    "skeleton_reuse_flag": row.get("skeleton_reuse_flag"),
                    "semantic_template_overlap_score": row.get("semantic_template_overlap_score"),
                    "slot_binding_diagnosis": row.get("slot_binding_diagnosis"),
                    "short_diagnosis": row.get("short_diagnosis"),
                }
            )
    return rows


def ensure_failure_file(out: Path, rows: list[dict[str, Any]]) -> None:
    path = out / "failure_case_samples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_aggregate_files(out: Path, seed_records: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_metrics(seed_records)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(
        out / "multi_seed_stability.json",
        {
            "schema_version": "chat_diversity_multi_seed_stability_v1",
            "all_seed_pass": aggregate["all_seed_pass"],
            "min_novel_response_rate": aggregate["min_novel_response_rate"],
            "max_template_copy_rate": aggregate["max_template_copy_rate"],
            "min_response_skeleton_diversity": aggregate["min_response_skeleton_diversity"],
            "max_response_skeleton_reuse_rate": aggregate["max_response_skeleton_reuse_rate"],
            "min_slot_binding_accuracy": aggregate["min_slot_binding_accuracy"],
            "min_finite_label_retention_accuracy": aggregate["min_finite_label_retention_accuracy"],
            "stddev_novel_response_rate": aggregate["stddev_novel_response_rate"],
            "stddev_template_copy_rate": aggregate["stddev_template_copy_rate"],
            "stddev_response_entropy": aggregate["stddev_response_entropy"],
        },
    )
    write_json(out / "novelty_aggregate.json", aggregate_numeric(seed_records, ["novel_response_rate", "template_copy_rate", "response_table_copy_rate", "semantic_template_overlap_rate", "slot_only_skeleton_reuse_rate"]))
    write_json(out / "skeleton_diversity_aggregate.json", aggregate_numeric(seed_records, ["response_skeleton_reuse_rate", "top_skeleton_rate", "response_skeleton_diversity"]))
    write_json(out / "vocabulary_entropy_aggregate.json", aggregate_numeric(seed_records, ["generated_to_train_vocab_ratio", "unique_bigram_count", "unique_trigram_count", "token_entropy", "response_entropy"]))
    write_json(out / "context_slot_aggregate.json", aggregate_numeric(seed_records, ["slot_binding_accuracy"]))
    write_json(out / "finite_label_retention_aggregate.json", aggregate_numeric(seed_records, ["finite_label_retention_accuracy"]))
    ensure_failure_file(out, failure_rows)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-080-root", type=Path, default=DEFAULT_UPSTREAM_080_ROOT)
    parser.add_argument("--upstream-079b-root", type=Path, default=DEFAULT_UPSTREAM_079B_ROOT)
    parser.add_argument("--upstream-079-root", type=Path, default=DEFAULT_UPSTREAM_079_ROOT)
    parser.add_argument("--upstream-078-root", type=Path, default=DEFAULT_UPSTREAM_078_ROOT)
    parser.add_argument("--upstream-074-root", type=Path, default=DEFAULT_UPSTREAM_074_ROOT)
    parser.add_argument("--seeds", default="2027,2028,2029")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    append_progress(out, "start", {"seeds": args.seeds})
    write_summary(out, "running", ["CHAT_DIVERSITY_MULTI_SEED_CONFIRM_RUNNING"], [])
    write_json(
        out / "queue.json",
        {
            "schema_version": "chat_diversity_multi_seed_confirm_queue_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM",
            "steps": ["validate_upstreams", "run_child_081_per_seed", "recheck_child_gates", "aggregate", "done"],
        },
    )
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as err:
        return fail(out, ["MULTI_SEED_CHAT_DIVERSITY_INSTABILITY_DETECTED"], str(err))

    missing = validate_upstreams(args)
    if missing:
        return fail(out, ["UPSTREAM_080_ARTIFACT_MISSING"], ", ".join(missing))

    upstream_080_summary = read_json(args.upstream_080_root / "summary.json")
    if "CHAT_COMPOSITION_DIVERSITY_REPAIR_POSITIVE" not in upstream_080_summary.get("verdicts", []):
        return fail(out, ["UPSTREAM_080_ARTIFACT_MISSING"], "080 positive verdict missing")

    write_json(
        out / "multi_seed_config.json",
        {
            "schema_version": "chat_diversity_multi_seed_confirm_config_v1",
            "multi_seed_eval_only": True,
            "train_step_count": 0,
            "seeds": seeds,
            "heartbeat_sec": args.heartbeat_sec,
            "no_mean_only_pass": True,
            "open_ended_generation_supported": False,
            "full_English_LM_supported": False,
            "language_grounding_claimed": False,
            "production_chat_claimed": False,
        },
    )
    write_json(
        out / "upstream_080_manifest.json",
        {
            "schema_version": "chat_diversity_multi_seed_upstream_080_manifest_v1",
            "checkpoint": str(checkpoint_path(args)),
            "upstream_080_root": str(args.upstream_080_root),
            "upstream_079b_root": str(args.upstream_079b_root),
            "upstream_079_root": str(args.upstream_079_root),
            "upstream_078_root": str(args.upstream_078_root),
            "upstream_074_root": str(args.upstream_074_root),
        },
    )

    seed_records: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    child_manifest: list[dict[str, Any]] = []
    for seed in seeds:
        child_out = out / f"seed_{seed}"
        if child_out.exists():
            shutil.rmtree(child_out)
        child_out.mkdir(parents=True, exist_ok=True)
        cmd = child_command(args, seed, child_out)
        child_started = time.time()
        code, elapsed = run_with_heartbeat(cmd, out, f"seed_{seed}", args.heartbeat_sec, out / "logs" / f"seed_{seed}.log")
        child_completed = time.time()
        record = seed_record(seed, child_out, cmd, code, elapsed, start, child_completed)
        seed_records.append(record)
        child_manifest.append(record)
        failure_rows.extend(collect_failure_rows(child_out, seed))
        append_jsonl(out / "seed_metrics.jsonl", record)
        write_json(out / "child_run_manifest.json", {"schema_version": "chat_diversity_child_run_manifest_v1", "children": child_manifest})
        write_aggregate_files(out, seed_records, failure_rows)
        partial_verdicts = final_verdicts(seed_records)
        write_summary(
            out,
            "running" if len(seed_records) < len(seeds) else ("passed" if partial_verdicts == POSITIVE_VERDICTS else "failed"),
            partial_verdicts,
            seed_records,
            {"partial_seed_count": len(seed_records)},
        )

    aggregate = write_aggregate_files(out, seed_records, failure_rows)
    verdicts = final_verdicts(seed_records)
    status = "passed" if verdicts == POSITIVE_VERDICTS else "failed"
    if not (out / "failure_case_samples.jsonl").exists():
        return fail(out, ["FAILURE_CASE_REPORT_MISSING"], "failure_case_samples.jsonl missing", seed_records)
    append_progress(out, "final", {"status": status, "verdicts": verdicts})
    write_summary(out, status, verdicts, seed_records, {"aggregate_metrics": aggregate})
    print(json.dumps(read_json(out / "summary.json"), sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
