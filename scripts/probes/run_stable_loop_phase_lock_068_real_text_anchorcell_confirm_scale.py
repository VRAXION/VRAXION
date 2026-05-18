#!/usr/bin/env python3
"""Orchestrator for STABLE_LOOP_PHASE_LOCK_068 real-text AnchorCell confirm scale."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINEWEB_ROOT = Path("S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B")
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm")
DEFAULT_FINEWEB_BYTES = 256 * 1024 * 1024
MAX_FINEWEB_BYTES = 1024 * 1024 * 1024
MAX_ANCHORCELL_EXAMPLES = 250_000
REQUIRED_SEEDS = [2026, 2027, 2028]

POSITIVE_VERDICTS = [
    "REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_POSITIVE",
    "FRESH_CHILD_RUNS_CONFIRMED",
    "CONFIRM_SNAPSHOT_IMMUTABILITY_PASSES",
    "CHILD_067_GATES_RECHECKED",
    "MULTI_SEED_MIN_GATE_PASSES",
    "CONFIRM_SCALE_LIMIT_ENFORCED",
    "FAILURE_CASE_REPORT_WRITTEN",
    "BASELINE_KNOCKOUT_STABLE",
    "CHECKPOINT_PIPELINE_MULTI_SEED_PASS",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
]

REQUIRED_CHILD_POSITIVE = "REAL_TEXT_ANCHORCELL_TRAINING_POC_POSITIVE"


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
    append_jsonl(
        out / "progress.jsonl",
        {
            "ts": utc_now(),
            "event": event,
            "payload": payload or {},
        },
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_snapshot(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": st.st_size,
        "modified_unix_ms": int(st.st_mtime_ns // 1_000_000),
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def disk_usage_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def write_report(out: Path, status: str, verdicts: list[str], seed_records: list[dict[str, Any]], message: str = "") -> None:
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_068_REAL_TEXT_ANCHORCELL_CONFIRM_SCALE Report",
        "",
        "068 confirms the 067 smoke result at controlled larger scale.",
        "This is not full 10B training, not production training, not a full English model, not language grounding, not GA, not public beta, and not hosted SaaS.",
        "No production performance claim, no training throughput claim, and no full-corpus claim is made.",
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
                "| seed | pass | heldout | ood | family_min | delta_vs_trigram | collapse | child_exit |",
                "| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for row in seed_records:
            mixed = row.get("mixed_arm", {})
            lines.append(
                "| {seed} | `{seed_pass}` | `{heldout:.3f}` | `{ood:.3f}` | `{family:.3f}` | `{tri:.3f}` | `{collapse}` | `{exit_code}` |".format(
                    seed=row.get("seed"),
                    seed_pass=row.get("seed_pass"),
                    heldout=float(mixed.get("heldout_exact_accuracy", 0.0)),
                    ood=float(mixed.get("ood_exact_accuracy", 0.0)),
                    family=float(mixed.get("family_min_accuracy", 0.0)),
                    tri=float(mixed.get("delta_vs_trigram", 0.0)),
                    collapse=mixed.get("collapse_detected"),
                    exit_code=row.get("child_exit_code"),
                )
            )
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(out: Path, status: str, verdicts: list[str], seed_records: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> None:
    payload = {
        "schema_version": "real_text_anchorcell_confirm_scale_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "seed_records": seed_records,
        "production_training_claimed": False,
        "production_performance_claimed": False,
        "training_throughput_claimed": False,
        "full_corpus_claimed": False,
        "public_beta_promoted": False,
        "hosted_saas_claimed": False,
    }
    if extra:
        payload.update(extra)
    write_json(out / "summary.json", payload)
    write_report(out, status, verdicts, seed_records)


def fail(out: Path, verdicts: list[str], message: str, seed_records: list[dict[str, Any]] | None = None) -> int:
    verdicts = ["REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_FAILS", *verdicts]
    records = seed_records or []
    append_progress(out, "failed", {"verdicts": verdicts, "message": message})
    write_summary(out, "failed", verdicts, records, {"message": message})
    write_report(out, "failed", verdicts, records, message)
    return 1


def run_with_heartbeat(
    cmd: list[str],
    out: Path,
    event_prefix: str,
    heartbeat_sec: int,
    log_path: Path,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    append_progress(out, f"{event_prefix}_started", {"command": command_string(cmd)})
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )
        last = started
        while True:
            code = proc.poll()
            now = time.time()
            if now - last >= heartbeat_sec:
                append_progress(
                    out,
                    f"{event_prefix}_heartbeat",
                    {
                        "pid": proc.pid,
                        "elapsed_sec": round(now - started, 3),
                        "log_path": str(log_path),
                    },
                )
                last = now
            if code is not None:
                elapsed = time.time() - started
                append_progress(
                    out,
                    f"{event_prefix}_completed",
                    {
                        "exit_code": code,
                        "elapsed_sec": round(elapsed, 3),
                        "log_path": str(log_path),
                    },
                )
                return code, elapsed
            time.sleep(min(2, max(1, heartbeat_sec // 5)))


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if seeds != REQUIRED_SEEDS:
        raise ValueError("068 requires exact seeds 2026,2027,2028; no mean-only or best-seed shortcut")
    return seeds


def validate_config(args: argparse.Namespace, seeds: list[int]) -> list[str]:
    failures: list[str] = []
    if args.fineweb_bytes > MAX_FINEWEB_BYTES:
        failures.append("CONFIRM_SCALE_LIMIT_EXCEEDED")
    if args.fineweb_bytes <= 50 * 1024 * 1024:
        failures.append("CONFIRM_SCALE_LIMIT_EXCEEDED")
    if args.anchorcell_examples > MAX_ANCHORCELL_EXAMPLES or args.anchorcell_examples <= 0:
        failures.append("CONFIRM_SCALE_LIMIT_EXCEEDED")
    if seeds != REQUIRED_SEEDS:
        failures.append("MULTI_SEED_INSTABILITY_DETECTED")
    return sorted(set(failures))


def find_confirm_source_file(fineweb_root: Path) -> Path:
    parquet_files = sorted(fineweb_root.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("no FineWeb parquet file available for bounded confirm snapshot extraction")
    return parquet_files[0]


def extract_parquet_text_pyarrow(source_file: Path, snapshot_path: Path, max_bytes: int, out: Path, heartbeat_sec: int) -> tuple[int, float, dict[str, Any]]:
    import pyarrow.parquet as pq

    started = time.time()
    last = started
    stats = {
        "source_file": str(source_file),
        "docs_emitted": 0,
        "docs_filtered": 0,
        "bytes_written": 0,
        "row_groups_opened": 0,
        "pyarrow_internal_extraction": True,
    }
    append_progress(
        out,
        "confirm_snapshot_extraction_started",
        {
            "command": f"internal_pyarrow_extract source={source_file} output={snapshot_path} max_bytes={max_bytes} max_files=1",
        },
    )
    parquet = pq.ParquetFile(source_file)
    names = set(parquet.schema_arrow.names)
    columns = [name for name in ["text", "language", "language_score", "score", "int_score"] if name in names]
    if "text" not in columns:
        raise RuntimeError(f"FineWeb parquet source has no text column: {source_file}")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("wb") as f:
        for batch in parquet.iter_batches(batch_size=1024, columns=columns):
            stats["row_groups_opened"] += 1
            arrays = {name: batch.column(idx).to_pylist() for idx, name in enumerate(batch.schema.names)}
            row_count = batch.num_rows
            for row_idx in range(row_count):
                if stats["bytes_written"] >= max_bytes:
                    break
                if "language" in arrays and arrays["language"][row_idx] != "en":
                    stats["docs_filtered"] += 1
                    continue
                if "language_score" in arrays and (arrays["language_score"][row_idx] is None or arrays["language_score"][row_idx] < 0.95):
                    stats["docs_filtered"] += 1
                    continue
                if "int_score" in arrays and arrays["int_score"][row_idx] is not None and arrays["int_score"][row_idx] < 3:
                    stats["docs_filtered"] += 1
                    continue
                text = arrays["text"][row_idx]
                if not text or not str(text).strip():
                    stats["docs_filtered"] += 1
                    continue
                encoded = (str(text).strip() + "\n").encode("utf-8", errors="ignore")
                remaining = max_bytes - stats["bytes_written"]
                f.write(encoded[:remaining])
                stats["bytes_written"] += min(len(encoded), remaining)
                stats["docs_emitted"] += 1
            now = time.time()
            if now - last >= heartbeat_sec:
                append_progress(
                    out,
                    "confirm_snapshot_extraction_heartbeat",
                    {
                        "elapsed_sec": round(now - started, 3),
                        "bytes_written": stats["bytes_written"],
                        "docs_emitted": stats["docs_emitted"],
                    },
                )
                last = now
            if stats["bytes_written"] >= max_bytes:
                break
    elapsed = time.time() - started
    append_progress(
        out,
        "confirm_snapshot_extraction_completed",
        {
            "exit_code": 0,
            "elapsed_sec": round(elapsed, 3),
            "bytes_written": stats["bytes_written"],
            "docs_emitted": stats["docs_emitted"],
        },
    )
    return 0, elapsed, stats


def build_or_reuse_snapshot(args: argparse.Namespace, out: Path, heartbeat_sec: int) -> tuple[Path, dict[str, Any], dict[str, Any], bool]:
    fineweb_root = Path(args.fineweb_root)
    source_file = find_confirm_source_file(fineweb_root)
    source_before = file_snapshot(source_file)
    snapshot_dir = out / "confirm_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"fineweb_confirm_{args.fineweb_bytes}.txt"
    extraction_command = f"internal_pyarrow_extract source={source_file} output={snapshot_path} max_bytes={args.fineweb_bytes} max_files=1"

    reused = snapshot_path.exists() and snapshot_path.stat().st_size >= args.fineweb_bytes
    if reused:
        append_progress(out, "confirm_snapshot_reused", {"snapshot": str(snapshot_path)})
        exit_code = 0
        elapsed = 0.0
        extraction_stats = {"reused_existing_snapshot": True}
    else:
        exit_code, elapsed, extraction_stats = extract_parquet_text_pyarrow(
            source_file,
            snapshot_path,
            args.fineweb_bytes,
            out,
            heartbeat_sec,
        )
    snapshot_before_children = file_snapshot(snapshot_path)
    if snapshot_before_children["size_bytes"] < int(args.fineweb_bytes * 0.95):
        raise RuntimeError(
            f"confirm snapshot too small: {snapshot_before_children['size_bytes']} bytes for requested {args.fineweb_bytes}"
        )
    source_after_extract = file_snapshot(source_file)
    source_mutated = source_before != source_after_extract
    source_manifest = {
        "schema_version": "fineweb_confirm_source_manifest_v1",
        "fineweb_root": str(fineweb_root),
        "source_files_actually_read": [source_before],
        "source_files_after_extraction": [source_after_extract],
        "fineweb_input_mutated_after_extraction": source_mutated,
        "full_parquet_sweep_used": False,
        "all_shard_training_used": False,
        "max_files": 1,
    }
    extraction_manifest = {
        "schema_version": "fineweb_extraction_manifest_v1",
        "snapshot_path": str(snapshot_path),
        "snapshot_hash_before_children": snapshot_before_children["sha256"],
        "snapshot_byte_count": snapshot_before_children["size_bytes"],
        "requested_byte_count": args.fineweb_bytes,
        "source_files": [source_before],
        "source_hashes": [source_before["sha256"]],
        "extraction_command": extraction_command,
        "extraction_timestamp": utc_now(),
        "extraction_exit_code": exit_code,
        "extraction_elapsed_sec": round(elapsed, 3),
        "reused_existing_snapshot": reused,
        "extraction_stats": extraction_stats,
    }
    write_json(out / "fineweb_confirm_source_manifest.json", source_manifest)
    write_json(out / "fineweb_extraction_manifest.json", extraction_manifest)
    return snapshot_path, source_manifest, extraction_manifest, source_mutated


def mixed_row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary.get("rows", [])
    for row in rows:
        if row.get("arm") == "MIXED_WITH_ROUTE_GRAMMAR_ON":
            return row
    for row in rows:
        if row.get("arm") == "MIXED_FINEWEB_ANCHORCELL_TRAINING":
            return row
    return {}


def parse_child(seed: int, child_out: Path, child_record: dict[str, Any], start_epoch: float) -> dict[str, Any]:
    summary_path = child_out / "summary.json"
    report_path = child_out / "report.md"
    baseline_path = child_out / "baseline_metrics.json"
    dataset_path = child_out / "dataset_manifest.json"
    collapse_path = child_out / "collapse_metrics.json"
    training_metrics_path = child_out / "training_metrics.jsonl"
    failure_samples_path = child_out / "inference_samples.jsonl"

    summary = read_json(summary_path) if summary_path.exists() else {}
    dataset = read_json(dataset_path) if dataset_path.exists() else {}
    baseline = read_json(baseline_path) if baseline_path.exists() else {}
    mixed = mixed_row_from_summary(summary)
    leakage = dataset.get("split_leakage_audit", {})
    verdicts = summary.get("verdicts", [])
    child_summary_newer = summary_path.exists() and summary_path.stat().st_mtime >= start_epoch
    child_report_newer = report_path.exists() and report_path.stat().st_mtime >= start_epoch
    baseline_eval_mismatch = bool(baseline.get("baseline_eval_mismatch", True))
    child_gate_recheck = all(
        [
            REQUIRED_CHILD_POSITIVE in verdicts,
            mixed.get("prediction_oracle_used") is False,
            baseline_eval_mismatch is False,
            leakage.get("train_eval_exact_input_overlap_count") == 0,
            leakage.get("train_ood_exact_input_overlap_count") == 0,
            mixed.get("checkpoint_save_load_pass") is True,
            mixed.get("rollback_success") is True,
            mixed.get("resume_from_checkpoint_pass") is True,
            mixed.get("collapse_detected") is False,
        ]
    )
    fresh = all(
        [
            child_record.get("child_run_started") is True,
            child_record.get("child_run_completed") is True,
            child_record.get("child_exit_code") == 0,
            child_summary_newer,
            child_report_newer,
            bool(child_record.get("child_command")),
        ]
    )

    failure_samples: list[dict[str, Any]] = []
    if float(mixed.get("family_min_accuracy", 0.0)) < 1.0:
        for sample in read_jsonl(failure_samples_path):
            if sample.get("arm") == "MIXED_WITH_ROUTE_GRAMMAR_ON" and not sample.get("correct", True):
                failure_samples.append(
                    {
                        "input": sample.get("input", ""),
                        "expected": sample.get("expected_output"),
                        "predicted": sample.get("predicted_output"),
                        "seed": seed,
                        "task_family": sample.get("task_family"),
                        "arm": sample.get("arm"),
                    }
                )

    training_rows = read_jsonl(training_metrics_path)
    child_gate = {
        "real_text_anchorcell_positive": REQUIRED_CHILD_POSITIVE in verdicts,
        "prediction_oracle_used": mixed.get("prediction_oracle_used"),
        "baseline_eval_mismatch": baseline_eval_mismatch,
        "train_eval_exact_input_overlap_count": leakage.get("train_eval_exact_input_overlap_count"),
        "train_ood_exact_input_overlap_count": leakage.get("train_ood_exact_input_overlap_count"),
        "checkpoint_save_load_pass": mixed.get("checkpoint_save_load_pass"),
        "rollback_success": mixed.get("rollback_success"),
        "resume_from_checkpoint_pass": mixed.get("resume_from_checkpoint_pass"),
        "collapse_detected": mixed.get("collapse_detected"),
    }
    seed_pass = fresh and child_gate_recheck
    return {
        **child_record,
        "seed": seed,
        "child_summary_newer_than_068_start": child_summary_newer,
        "child_report_newer_than_068_start": child_report_newer,
        "fresh_child_artifacts": fresh,
        "child_067_gate_recheck_pass": child_gate_recheck,
        "child_gate_recheck": child_gate,
        "seed_pass": seed_pass,
        "mixed_arm": mixed,
        "dataset_counts": {
            "train_examples": dataset.get("train_examples"),
            "heldout_examples": dataset.get("heldout_examples"),
            "ood_examples": dataset.get("ood_examples"),
            "data_mix": dataset.get("data_mix", {}),
        },
        "child_verdicts": verdicts,
        "collapse_metrics_present": collapse_path.exists(),
        "training_rows": training_rows,
        "failure_samples": failure_samples,
        "child_out": str(child_out),
    }


def aggregate(seed_records: list[dict[str, Any]], out: Path, start_epoch: float, snapshot_path: Path, snapshot_before_hash: str, source_manifest: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    deltas_uni = [float(r.get("mixed_arm", {}).get("delta_vs_unigram", 0.0)) for r in seed_records]
    deltas_bi = [float(r.get("mixed_arm", {}).get("delta_vs_bigram", 0.0)) for r in seed_records]
    deltas_tri = [float(r.get("mixed_arm", {}).get("delta_vs_trigram", 0.0)) for r in seed_records]
    seed_passes = [bool(r.get("seed_pass")) for r in seed_records]
    snapshot_after = file_snapshot(snapshot_path)
    snapshot_unchanged = snapshot_after["sha256"] == snapshot_before_hash
    source_after = [file_snapshot(Path(row["path"])) for row in source_manifest["source_files_actually_read"]]
    source_unchanged = all(
        before["sha256"] == after["sha256"]
        and before["size_bytes"] == after["size_bytes"]
        and before["modified_unix_ms"] == after["modified_unix_ms"]
        for before, after in zip(source_manifest["source_files_actually_read"], source_after)
    )
    per_seed_metrics = {
        "per_seed_delta_vs_unigram": dict(zip([str(r["seed"]) for r in seed_records], deltas_uni)),
        "per_seed_delta_vs_bigram": dict(zip([str(r["seed"]) for r in seed_records], deltas_bi)),
        "per_seed_delta_vs_trigram": dict(zip([str(r["seed"]) for r in seed_records], deltas_tri)),
        "min_delta_vs_unigram": min(deltas_uni) if deltas_uni else 0.0,
        "min_delta_vs_bigram": min(deltas_bi) if deltas_bi else 0.0,
        "min_delta_vs_trigram": min(deltas_tri) if deltas_tri else 0.0,
        "stddev_delta_vs_unigram": statistics.pstdev(deltas_uni) if len(deltas_uni) > 1 else 0.0,
        "stddev_delta_vs_bigram": statistics.pstdev(deltas_bi) if len(deltas_bi) > 1 else 0.0,
        "stddev_delta_vs_trigram": statistics.pstdev(deltas_tri) if len(deltas_tri) > 1 else 0.0,
        "heldout_exact_accuracy": {str(r["seed"]): r.get("mixed_arm", {}).get("heldout_exact_accuracy") for r in seed_records},
        "ood_exact_accuracy": {str(r["seed"]): r.get("mixed_arm", {}).get("ood_exact_accuracy") for r in seed_records},
        "context_carry_accuracy": {str(r["seed"]): r.get("mixed_arm", {}).get("context_carry_accuracy") for r in seed_records},
        "paired_counterfactual_accuracy": {str(r["seed"]): r.get("mixed_arm", {}).get("paired_counterfactual_accuracy") for r in seed_records},
        "family_min_accuracy": {str(r["seed"]): r.get("mixed_arm", {}).get("family_min_accuracy") for r in seed_records},
        "collapse_metrics": {str(r["seed"]): {
            "top_output_rate": r.get("mixed_arm", {}).get("top_output_rate"),
            "space_output_rate": r.get("mixed_arm", {}).get("space_output_rate"),
            "empty_output_rate": r.get("mixed_arm", {}).get("empty_output_rate"),
            "collapse_detected": r.get("mixed_arm", {}).get("collapse_detected"),
        } for r in seed_records},
    }
    baseline_stable = (
        per_seed_metrics["min_delta_vs_unigram"] > 0.10
        and per_seed_metrics["min_delta_vs_bigram"] > 0.05
        and per_seed_metrics["min_delta_vs_trigram"] > 0.03
    )
    checkpoint_multi_seed = all(
        r.get("mixed_arm", {}).get("checkpoint_save_load_pass") is True
        and r.get("mixed_arm", {}).get("rollback_success") is True
        and r.get("mixed_arm", {}).get("resume_from_checkpoint_pass") is True
        and r.get("mixed_arm", {}).get("eval_after_reload_matches_before") is True
        for r in seed_records
    )
    total_elapsed = time.time() - start_epoch
    first_mix = seed_records[0].get("dataset_counts", {}).get("data_mix", {}) if seed_records else {}
    actual_synthetic_examples = sum(
        int(v)
        for k, v in first_mix.items()
        if k != "FINEWEB_RAW_CONTINUATION" and isinstance(v, int)
    )
    budget = {
        "elapsed_sec_per_seed": {str(r["seed"]): r.get("child_elapsed_sec") for r in seed_records},
        "total_elapsed_sec": round(total_elapsed, 3),
        "peak_memory_available": False,
        "peak_memory": None,
        "disk_usage_under_out_dir_bytes": disk_usage_bytes(out),
        "fineweb_bytes_used": args.fineweb_bytes,
        "configured_anchorcell_examples": args.anchorcell_examples,
        "synthetic_examples_used": actual_synthetic_examples,
        "no_production_performance_claim": True,
        "no_training_throughput_claim": True,
        "no_full_corpus_claim": True,
    }
    multi_seed_pass = len(seed_records) == 3 and all(seed_passes)
    positive = all([multi_seed_pass, snapshot_unchanged, source_unchanged, baseline_stable, checkpoint_multi_seed])
    if positive:
        verdicts = POSITIVE_VERDICTS.copy()
    else:
        verdicts = ["REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_FAILS"]
        if not multi_seed_pass:
            verdicts.append("MULTI_SEED_INSTABILITY_DETECTED")
        if any(not r.get("fresh_child_artifacts") for r in seed_records):
            verdicts.append("STALE_CHILD_ARTIFACT_USED")
        if not snapshot_unchanged:
            verdicts.append("CONFIRM_SNAPSHOT_MUTATION_DETECTED")
        if not source_unchanged:
            verdicts.append("FINEWEB_INPUT_MUTATION_DETECTED")
        if any(not r.get("child_067_gate_recheck_pass") for r in seed_records):
            verdicts.append("CHILD_GATE_RECHECK_FAILS")
        if not baseline_stable:
            verdicts.append("BASELINE_KNOCKOUT_REGRESSION")
        if not checkpoint_multi_seed:
            verdicts.append("CHECKPOINT_RELOAD_FAILS")
    details = {
        "snapshot_after_children": snapshot_after,
        "snapshot_unchanged": snapshot_unchanged,
        "source_files_after_all_runs": source_after,
        "fineweb_input_unchanged": source_unchanged,
        "baseline_stable": baseline_stable,
        "checkpoint_multi_seed": checkpoint_multi_seed,
        "budget": budget,
        "aggregate_metrics": per_seed_metrics,
    }
    return sorted(set(verdicts), key=verdicts.index), details


def write_aggregate_artifacts(out: Path, seed_records: list[dict[str, Any]], verdicts: list[str], details: dict[str, Any], args: argparse.Namespace) -> None:
    for row in seed_records:
        append_jsonl(out / "seed_metrics.jsonl", {
            "seed": row["seed"],
            "seed_pass": row["seed_pass"],
            "fresh_child_artifacts": row["fresh_child_artifacts"],
            "child_067_gate_recheck_pass": row["child_067_gate_recheck_pass"],
            "mixed_arm": row["mixed_arm"],
        })
    write_json(out / "aggregate_metrics.json", details["aggregate_metrics"])
    write_json(out / "multi_seed_stability.json", {
        "all_three_seeds_required": True,
        "mean_only_pass_allowed": False,
        "best_seed_pass_allowed": False,
        "two_of_three_pass_allowed": False,
        "seed_passes": {str(r["seed"]): r["seed_pass"] for r in seed_records},
        "multi_seed_min_gate_pass": all(r["seed_pass"] for r in seed_records),
    })
    write_json(out / "training_curve_report.json", {
        "schema_version": "training_curve_report_v1",
        "rows": [
            {
                "seed": r["seed"],
                "training_rows": r.get("training_rows", []),
                "elapsed_sec": r.get("child_elapsed_sec"),
            }
            for r in seed_records
        ],
        "no_production_performance_claim": True,
        "no_training_throughput_claim": True,
    })
    write_json(out / "baseline_knockout_aggregate.json", {
        "schema_version": "baseline_knockout_aggregate_v1",
        "per_seed_delta_vs_unigram": details["aggregate_metrics"]["per_seed_delta_vs_unigram"],
        "per_seed_delta_vs_bigram": details["aggregate_metrics"]["per_seed_delta_vs_bigram"],
        "per_seed_delta_vs_trigram": details["aggregate_metrics"]["per_seed_delta_vs_trigram"],
        "min_delta_vs_unigram": details["aggregate_metrics"]["min_delta_vs_unigram"],
        "min_delta_vs_bigram": details["aggregate_metrics"]["min_delta_vs_bigram"],
        "min_delta_vs_trigram": details["aggregate_metrics"]["min_delta_vs_trigram"],
        "stddev_delta_vs_unigram": details["aggregate_metrics"]["stddev_delta_vs_unigram"],
        "stddev_delta_vs_bigram": details["aggregate_metrics"]["stddev_delta_vs_bigram"],
        "stddev_delta_vs_trigram": details["aggregate_metrics"]["stddev_delta_vs_trigram"],
    })
    failure_path = out / "failure_case_samples.jsonl"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text("", encoding="utf-8")
    for row in seed_records:
        for sample in row.get("failure_samples", []):
            append_jsonl(failure_path, sample)
    write_json(out / "checkpoint_pipeline_report.json", {
        "schema_version": "checkpoint_pipeline_report_v1",
        "per_seed": {
            str(r["seed"]): {
                "checkpoint_save_load_pass": r.get("mixed_arm", {}).get("checkpoint_save_load_pass"),
                "eval_after_reload_matches_before": r.get("mixed_arm", {}).get("eval_after_reload_matches_before"),
                "rollback_success": r.get("mixed_arm", {}).get("rollback_success"),
                "resume_from_checkpoint_pass": r.get("mixed_arm", {}).get("resume_from_checkpoint_pass"),
                "resumed_checkpoint_hash_changed": r.get("mixed_arm", {}).get("resumed_checkpoint_hash_changed"),
            }
            for r in seed_records
        },
        "multi_seed_checkpoint_pipeline_pass": details["checkpoint_multi_seed"],
    })
    write_json(out / "confirm_config.json", {
        "schema_version": "real_text_anchorcell_confirm_config_v1",
        "fineweb_root": str(args.fineweb_root),
        "fineweb_bytes": args.fineweb_bytes,
        "anchorcell_examples": args.anchorcell_examples,
        "seeds": REQUIRED_SEEDS,
        "mode": "confirm",
        "fineweb_bytes_limit": MAX_FINEWEB_BYTES,
        "anchorcell_examples_limit": MAX_ANCHORCELL_EXAMPLES,
        "full_corpus_training_attempted": False,
    })
    write_summary(out, "done" if "REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_POSITIVE" in verdicts else "failed", verdicts, seed_records, details)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fineweb-root", type=Path, default=DEFAULT_FINEWEB_ROOT)
    parser.add_argument("--fineweb-bytes", type=int, default=DEFAULT_FINEWEB_BYTES)
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--anchorcell-examples", type=int, default=100_000)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["progress.jsonl", "seed_metrics.jsonl", "failure_case_samples.jsonl"]:
        path = out / name
        if path.exists():
            path.unlink()
    start_epoch = time.time()
    start_iso = utc_now()
    append_progress(out, "start", {"start_iso": start_iso})
    write_summary(out, "running", [], [], {"started_at": start_iso})

    try:
        seeds = parse_seeds(args.seeds)
    except Exception as exc:
        return fail(out, ["MULTI_SEED_INSTABILITY_DETECTED"], str(exc))

    failures = validate_config(args, seeds)
    if failures:
        return fail(out, failures, "confirm scale limits or seed requirements failed before side effects")
    append_progress(out, "config_loaded", {"fineweb_bytes": args.fineweb_bytes, "seeds": seeds, "anchorcell_examples": args.anchorcell_examples})

    write_json(out / "queue.json", {
        "schema_version": "real_text_anchorcell_confirm_queue_v1",
        "probe": "STABLE_LOOP_PHASE_LOCK_068_REAL_TEXT_ANCHORCELL_CONFIRM_SCALE",
        "seeds": seeds,
        "fineweb_bytes": args.fineweb_bytes,
        "anchorcell_examples": args.anchorcell_examples,
        "heartbeat_sec": args.heartbeat_sec,
        "production_training_claimed": False,
        "public_beta_promoted": False,
        "hosted_saas_claimed": False,
    })

    try:
        snapshot_path, source_manifest, extraction_manifest, source_mutated_after_extract = build_or_reuse_snapshot(args, out, args.heartbeat_sec)
    except Exception as exc:
        verdict = "CONFIRM_SCALE_LIMIT_EXCEEDED" if "snapshot too small" in str(exc) else "FINEWEB_SMOKE_SOURCE_MISSING"
        return fail(out, [verdict], f"confirm snapshot creation failed: {exc}")
    if source_mutated_after_extract:
        return fail(out, ["FINEWEB_INPUT_MUTATION_DETECTED"], "FineWeb input changed during snapshot extraction")
    snapshot_before_hash = extraction_manifest["snapshot_hash_before_children"]

    seed_records: list[dict[str, Any]] = []
    for seed in seeds:
        child_out = out / f"seed_{seed}"
        child_cmd = [
            "cargo",
            "run",
            "--release",
            "-p",
            "instnct-core",
            "--example",
            "phase_lane_real_text_anchorcell_training_poc",
            "--",
            "--out",
            str(child_out),
            "--fineweb-root",
            str(args.fineweb_root),
            "--fineweb-source",
            str(snapshot_path),
            "--mode",
            "confirm",
            "--seed",
            str(seed),
            "--heartbeat-sec",
            str(args.heartbeat_sec),
            "--fineweb-bytes",
            str(args.fineweb_bytes),
            "--anchorcell-examples",
            str(args.anchorcell_examples),
        ]
        record = {
            "seed": seed,
            "child_run_started": True,
            "child_command": command_string(child_cmd),
        }
        code, elapsed = run_with_heartbeat(
            child_cmd,
            out,
            f"child_seed_{seed}",
            args.heartbeat_sec,
            out / "logs" / f"child_seed_{seed}.log",
        )
        record.update({
            "child_run_completed": True,
            "child_exit_code": code,
            "child_elapsed_sec": round(elapsed, 3),
        })
        parsed = parse_child(seed, child_out, record, start_epoch)
        seed_records.append(parsed)
        write_summary(out, "running", [], seed_records, {"latest_seed": seed})

    verdicts, details = aggregate(seed_records, out, start_epoch, snapshot_path, snapshot_before_hash, source_manifest, args)
    write_aggregate_artifacts(out, seed_records, verdicts, details, args)
    append_progress(out, "aggregate_completed", {"verdicts": verdicts})
    append_progress(out, "done", {"verdicts": verdicts, "elapsed_sec": round(time.time() - start_epoch, 3)})
    if "REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_POSITIVE" in verdicts:
        print("068 complete: " + ",".join(verdicts))
        return 0
    print("068 completed with failure verdicts: " + ",".join(verdicts))
    return 1


if __name__ == "__main__":
    sys.exit(main())
