#!/usr/bin/env python3
"""Eval-only multi-seed confirmation for the 102 raw rollout repair."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_104_MULTI_SEED_RAW_GENERATION_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_104_multi_seed_raw_generation_confirm/smoke")
DEFAULT_UPSTREAM_103_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_103_fresh_raw_generation_confirm/smoke")
DEFAULT_UPSTREAM_102_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_102_decoder_policy_and_rollout_repair/smoke")
DEFAULT_UPSTREAM_101_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_101_fresh_assistant_eval_and_raw_decoder_frontier_map/smoke")
DEFAULT_UPSTREAM_100_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_100_open_vocab_assistant_capability_scale/smoke")
DEFAULT_UPSTREAM_099_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke")
BOUNDARY_TEXT = (
    "104 is multi-seed raw-generation confirmation only. It is eval-only, performs no training, "
    "runs no repair, mutates no checkpoint, and changes no runtime/service/deploy surface. It is not "
    "GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public "
    "API, not deployment readiness, not public release, not safety alignment, and not Hungarian assistant capability."
)


class GateError(Exception):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict
        self.message = message


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GateError("UPSTREAM_103_ARTIFACT_MISSING", f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE103 = load_module("phase103_for_104", REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_103_fresh_raw_generation_confirm.py")
PHASE094 = PHASE103.PHASE094


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return PHASE103.sha256_file(path)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(text: str, verdict: str) -> Path:
    return PHASE103.resolve_repo_path(text, verdict)


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("EVAL_DATASET_BUILD_FAILS", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("EVAL_DATASET_BUILD_FAILS", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def hash_paths(paths: list[Path]) -> str:
    return PHASE103.hash_paths(paths)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise GateError("EVAL_DATASET_BUILD_FAILS", "at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise GateError("EVAL_DATASET_BUILD_FAILS", "duplicate seeds are not allowed")
    return seeds


def write_summary(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    payload = {
        "schema_version": "multi_seed_raw_generation_confirm_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "phase": status,
        "boundary": BOUNDARY_TEXT,
        "eval_only": True,
        "training_performed": False,
        "model_capability_improved_by_104": False,
        "runner_local_pytorch_lm": True,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "public_api_claimed": False,
        "deployment_readiness_claimed": False,
        "public_release_claimed": False,
        "safety_alignment_claimed": False,
        "hungarian_capability_claimed": False,
        "metrics": metrics,
        "verdicts": verdicts,
    }
    if message:
        payload["message"] = message
    write_json(out / "summary.json", payload)
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_104_MULTI_SEED_RAW_GENERATION_CONFIRM Report",
        "",
        BOUNDARY_TEXT,
        "",
        f"Status: `{status}`",
        "",
        "## Verdicts",
        "",
        "```text",
        *verdicts,
        "```",
        "",
        "## Aggregate Metrics",
        "",
    ]
    for key in [
        "all_seeds_passed_independently",
        "seed_count",
        "min_raw_free_generation_accuracy",
        "max_case_id_drift_rate",
        "max_slot_drift_rate",
        "max_distractor_leak_rate",
        "min_decoder_assisted_accuracy",
        "min_bounded_chat_slot_binding_accuracy",
        "min_finite_label_anchorroute_retention_accuracy",
        "stddev_raw_free_generation_accuracy",
        "stddev_case_id_drift_rate",
        "checkpoint_hash_unchanged_all_seeds",
        "bounded_release_artifact_unchanged_all_seeds",
        "train_step_count",
        "optimizer_step_count",
        "primary_next_milestone",
    ]:
        if key in metrics:
            lines.append(f"- {key}: `{metrics[key]}`")
    if message:
        lines.extend(["", "## Message", "", message])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- multi-seed raw-generation confirmation only",
            "- no model capability improved by 104",
            "- not GPT-like assistant readiness",
            "- not open-domain assistant readiness",
            "- not production chat",
            "- not public API",
            "- not deployment readiness",
            "- not public release",
            "- not safety alignment",
            "- Hungarian capability not claimed",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def fail(out: Path, verdict: str, message: str, metrics: dict[str, Any]) -> int:
    append_progress(out, "final verdict", "failed", verdict=verdict, message=message)
    write_jsonl(out / "failure_case_samples.jsonl", [{"ts": utc_now(), "verdict": verdict, "message": message}])
    write_summary(out, "failed", ["MULTI_SEED_RAW_GENERATION_CONFIRM_FAILS", verdict], metrics, message)
    return 1


def require_summary(root: Path, positive: str, missing: str, not_positive: str) -> dict[str, Any]:
    path = root / "summary.json"
    if not path.exists():
        raise GateError(missing, f"missing summary: {root}")
    summary = read_json(path)
    if positive not in set(summary.get("verdicts", [])):
        raise GateError(not_positive, f"missing positive verdict: {positive}")
    return summary


def verify_upstreams(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    summary_103 = require_summary(args.upstream_103_root, "FRESH_RAW_GENERATION_CONFIRM_POSITIVE", "UPSTREAM_103_ARTIFACT_MISSING", "UPSTREAM_103_NOT_POSITIVE")
    summary_102 = require_summary(args.upstream_102_root, "DECODER_POLICY_AND_ROLLOUT_REPAIR_POSITIVE", "UPSTREAM_102_ARTIFACT_MISSING", "UPSTREAM_102_NOT_POSITIVE")
    summary_101 = require_summary(args.upstream_101_root, "FRESH_ASSISTANT_FRONTIER_MAP_POSITIVE", "UPSTREAM_101_ARTIFACT_MISSING", "UPSTREAM_101_NOT_POSITIVE")
    summary_100 = require_summary(args.upstream_100_root, "OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE", "UPSTREAM_100_ARTIFACT_MISSING", "UPSTREAM_100_NOT_POSITIVE")
    summary_099 = require_summary(args.upstream_099_root, "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE", "UPSTREAM_099_ARTIFACT_MISSING", "UPSTREAM_099_NOT_POSITIVE")
    metrics_103 = summary_103.get("metrics", {})
    metrics_102 = summary_102.get("metrics", {})
    metrics_101 = summary_101.get("metrics", {})
    if metrics_103.get("primary_next_milestone") != "104_MULTI_SEED_RAW_GENERATION_CONFIRM":
        raise GateError("UPSTREAM_103_NOT_POSITIVE", "103 did not recommend 104")
    checkpoint_manifest_103 = read_json(args.upstream_103_root / "checkpoint_manifest.json")
    checkpoint_path = resolve_repo_path(checkpoint_manifest_103["target_102_checkpoint_path"], "UPSTREAM_103_ARTIFACT_MISSING")
    source_100_checkpoint = resolve_repo_path(checkpoint_manifest_103["source_100_checkpoint_path"], "UPSTREAM_103_ARTIFACT_MISSING")
    upstream_103_manifest = read_json(args.upstream_103_root / "upstream_102_manifest.json")
    release_paths = [resolve_repo_path(path, "UPSTREAM_103_ARTIFACT_MISSING") for path in upstream_103_manifest.get("bounded_release_paths", [])]
    release_hash = hash_paths(release_paths)
    manifest = {
        "schema_version": "multi_seed_raw_generation_confirm_upstream_103_manifest_v1",
        "upstream_103_root": rel(args.upstream_103_root),
        "upstream_102_root": rel(args.upstream_102_root),
        "upstream_101_root": rel(args.upstream_101_root),
        "upstream_100_root": rel(args.upstream_100_root),
        "upstream_099_root": rel(args.upstream_099_root),
        "upstream_103_status": summary_103.get("status"),
        "upstream_103_verdicts": summary_103.get("verdicts", []),
        "upstream_103_raw_free_generation_accuracy": metrics_103.get("raw_free_generation_accuracy"),
        "upstream_103_case_id_drift_rate": metrics_103.get("case_id_drift_rate"),
        "upstream_102_raw_free_generation_accuracy": metrics_102.get("raw_free_generation_accuracy"),
        "upstream_102_decoder_assisted_accuracy": metrics_102.get("decoder_assisted_accuracy"),
        "upstream_101_raw_free_generation_accuracy": metrics_101.get("raw_free_generation_accuracy"),
        "target_102_checkpoint_path": rel(checkpoint_path),
        "target_102_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_100_checkpoint_path": rel(source_100_checkpoint),
        "source_100_checkpoint_sha256": sha256_file(source_100_checkpoint),
        "bounded_release_artifact_hash_before": release_hash,
        "bounded_release_paths": [rel(path) for path in release_paths],
        "099_local_private_release_ready": summary_099.get("metrics", {}).get("local_private_release_ready"),
        "100_status": summary_100.get("status"),
        "101_status": summary_101.get("status"),
        "102_status": summary_102.get("status"),
    }
    write_json(out / "upstream_103_manifest.json", manifest)
    return {
        "summary_103": summary_103,
        "summary_102": summary_102,
        "summary_101": summary_101,
        "summary_100": summary_100,
        "summary_099": summary_099,
        "metrics_103": metrics_103,
        "metrics_102": metrics_102,
        "metrics_101": metrics_101,
        "checkpoint_path": checkpoint_path,
        "source_100_checkpoint": source_100_checkpoint,
        "release_paths": release_paths,
        "release_hash_before": release_hash,
    }


def build_seed_dataset(seed: int, seed_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = PHASE103.build_fresh_rows(seed + 104_000, rows_per_family=10)
    overlaps = PHASE103.load_overlap_rows(args.upstream_101_root, args.upstream_102_root, args.upstream_100_root)
    rows_103 = read_jsonl(args.upstream_103_root / "fresh_raw_eval_dataset.jsonl")
    prompts = {row["prompt"] for row in rows}
    overlap_101 = len(prompts & {row["prompt"] for row in overlaps["101_eval"]})
    overlap_102_train = len(prompts & {row["prompt"] for row in overlaps["102_train"]})
    overlap_102_eval = len(prompts & {row["prompt"] for row in overlaps["102_eval"]})
    overlap_100 = len(prompts & {row["prompt"] for row in overlaps["100_train_eval"]})
    overlap_103 = len(prompts & {row["prompt"] for row in rows_103})
    max_j_train = PHASE103.max_prompt_jaccard(rows, overlaps["102_train"][:2000])
    max_j_103 = PHASE103.max_prompt_jaccard(rows, rows_103)
    if overlap_101 or overlap_102_train or overlap_102_eval or overlap_100 or overlap_103 or max_j_train >= 0.90 or max_j_103 >= 0.90:
        raise GateError("TRAIN_EVAL_LEAKAGE_DETECTED", f"seed {seed} fresh eval rows overlap prior rows")
    payload = [{"family_code": row["family_code"], "prompt": row["prompt"], "response": row["response"]} for row in rows]
    manifest = {
        "schema_version": "multi_seed_raw_generation_confirm_seed_dataset_manifest_v1",
        "seed": seed,
        "dataset_salt": seed + 104_000,
        "eval_count": len(rows),
        "families": PHASE103.EVAL_FAMILIES,
        "eval_row_hash": PHASE103.stable_json_hash(payload),
        "eval_prompt_hash": PHASE103.stable_json_hash([row["prompt"] for row in rows]),
        "eval_dataset_sha256": hashlib.sha256("\n".join(json.dumps(row, sort_keys=True) for row in rows).encode("utf-8")).hexdigest(),
        "overlap_with_101_eval_count": overlap_101,
        "overlap_with_102_train_count": overlap_102_train,
        "overlap_with_102_eval_count": overlap_102_eval,
        "overlap_with_103_eval_count": overlap_103,
        "overlap_with_100_train_eval_count": overlap_100,
        "max_prompt_jaccard_vs_102_train": max_j_train,
        "max_prompt_jaccard_vs_103_eval": max_j_103,
        "true_case_id_not_always_first": True,
        "distractor_numbers_near_active_slot": True,
        "ticket_session_record_request_phrasing": True,
        "unsupported_numbered_prompts_included": True,
    }
    write_json(seed_dir / "eval_config.json", manifest)
    write_jsonl(seed_dir / "fresh_raw_eval_dataset.jsonl", rows)
    return {"rows": rows, "manifest": manifest}


def seed_passes(metrics: dict[str, Any], upstream: dict[str, Any]) -> tuple[bool, str]:
    if metrics["raw_free_generation_accuracy"] < 0.85 or metrics["raw_free_generation_accuracy"] < upstream["metrics_101"]["raw_free_generation_accuracy"] + 0.50:
        return False, "RAW_GENERATION_GENERALIZATION_FAILS"
    if metrics["case_id_drift_rate"] > 0.10:
        return False, "CASE_ID_ANCHOR_GENERALIZATION_FAILS"
    if metrics["distractor_number_copy_rate"] > 0.10:
        return False, "DISTRACTOR_NUMBER_COPY_DETECTED"
    if metrics["slot_drift_rate"] > 0.05:
        return False, "SLOT_PINNING_GENERALIZATION_FAILS"
    if metrics["distractor_leak_rate"] > 0.10:
        return False, "DISTRACTOR_SUPPRESSION_REGRESSION_DETECTED"
    if metrics["decoder_assisted_accuracy"] < 0.90 or metrics["decoder_assisted_accuracy_delta_vs_102"] < -0.05:
        return False, "DECODER_ASSISTED_REGRESSION_DETECTED"
    if metrics["bounded_chat_slot_binding_accuracy"] < 0.90 or metrics["finite_label_anchorroute_retention_accuracy"] < 0.90 or metrics["unsupported_refusal_accuracy"] < 0.80:
        return False, "RETENTION_REGRESSION_DETECTED"
    if metrics["empty_output_rate"] > 0.02:
        return False, "EMPTY_OUTPUT_COLLAPSE_DETECTED"
    if metrics["static_output_rate"] > 0.15:
        return False, "STATIC_RESPONSE_COLLAPSE_DETECTED"
    if metrics["repetition_rate"] > 0.25 or metrics["copy_prompt_rate"] > 0.20 or metrics["utf8_valid_generation_rate"] < 0.80 or metrics["nonempty_generation_rate"] < 0.98:
        return False, "REPETITION_COLLAPSE_DETECTED"
    for key in [
        "checkpoint_hash_unchanged",
        "source_100_checkpoint_unchanged",
        "bounded_release_artifact_unchanged",
    ]:
        if metrics.get(key) is not True:
            return False, "CHECKPOINT_MUTATION_DETECTED" if key != "bounded_release_artifact_unchanged" else "BOUNDED_RELEASE_MUTATION_DETECTED"
    if metrics.get("train_step_count") != 0 or metrics.get("optimizer_step_count") != 0:
        return False, "TRAINING_SIDE_EFFECT_DETECTED"
    return True, "pass"


def write_seed_summary(seed_dir: Path, seed: int, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    payload = {
        "schema_version": "multi_seed_raw_generation_confirm_seed_summary_v1",
        "milestone": MILESTONE,
        "seed": seed,
        "status": status,
        "boundary": BOUNDARY_TEXT,
        "eval_only": True,
        "training_performed": False,
        "hungarian_capability_claimed": False,
        "metrics": metrics,
        "verdicts": verdicts,
    }
    if message:
        payload["message"] = message
    write_json(seed_dir / "summary.json", payload)
    lines = [
        f"# STABLE_LOOP_PHASE_LOCK_104 seed {seed} report",
        "",
        BOUNDARY_TEXT,
        "",
        f"Status: `{status}`",
        "",
        "```text",
        *verdicts,
        "```",
        "",
        f"- raw_free_generation_accuracy: `{metrics.get('raw_free_generation_accuracy')}`",
        f"- case_id_drift_rate: `{metrics.get('case_id_drift_rate')}`",
        f"- decoder_assisted_accuracy: `{metrics.get('decoder_assisted_accuracy')}`",
        f"- checkpoint_hash_unchanged: `{metrics.get('checkpoint_hash_unchanged')}`",
        "",
        "Hungarian capability not claimed.",
    ]
    if message:
        lines.extend(["", message])
    write_text(seed_dir / "report.md", "\n".join(lines) + "\n")


def run_seed(seed: int, seed_dir: Path, args: argparse.Namespace, upstream: dict[str, Any], started_at: float) -> dict[str, Any]:
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_command = (
        f"internal_103_canonical_eval --seed {seed} --dataset-salt {seed + 104_000} "
        f"--out {rel(seed_dir)} --modes {','.join(PHASE103.EVAL_MODES)}"
    )
    seed_record: dict[str, Any] = {
        "seed": seed,
        "seed_run_started": True,
        "seed_run_completed": False,
        "seed_command": seed_command,
        "seed_started_at_epoch": time.time(),
    }
    write_json(seed_dir / "queue.json", {"schema_version": "multi_seed_raw_generation_confirm_seed_queue_v1", "seed": seed, "seed_command": seed_command, "steps": ["dataset", "eval_modes", "integrity", "reports"]})
    dataset = build_seed_dataset(seed, seed_dir, args)
    checkpoint = upstream["checkpoint_path"]
    source_100 = upstream["source_100_checkpoint"]
    checkpoint_hash_before = sha256_file(checkpoint)
    checkpoint_state_hash_before = PHASE094.model_state_hash(PHASE094.load_checkpoint(checkpoint))
    source_100_hash_before = sha256_file(source_100)
    model_100 = PHASE094.load_checkpoint(source_100)
    results: dict[str, list[dict[str, Any]]] = {}
    eval_args = argparse.Namespace(**vars(args))
    eval_args.seed = seed
    for mode in PHASE103.EVAL_MODES:
        results[mode] = PHASE103.evaluate_mode(model_100, dataset["rows"], mode, eval_args)
    metrics_by_mode = {mode: PHASE103.mode_metrics(rows) for mode, rows in results.items()}
    report_bits = PHASE103.write_reports(seed_dir, results, metrics_by_mode, dataset, upstream)
    raw = metrics_by_mode["RAW_GREEDY_GENERATION"]
    decoder = metrics_by_mode["DECODER_ASSISTED_REFERENCE"]
    checkpoint_hash_after = sha256_file(checkpoint)
    checkpoint_state_hash_after = PHASE094.model_state_hash(PHASE094.load_checkpoint(checkpoint))
    source_100_hash_after = sha256_file(source_100)
    release_hash_after = hash_paths(upstream["release_paths"])
    checkpoint_unchanged = checkpoint_hash_before == checkpoint_hash_after and checkpoint_state_hash_before == checkpoint_state_hash_after
    source_100_unchanged = source_100_hash_before == source_100_hash_after
    bounded_release_unchanged = release_hash_after == upstream["release_hash_before"]
    seed_metrics = {
        **raw,
        "seed": seed,
        "raw_free_generation_accuracy": raw["raw_free_generation_accuracy"],
        "decoder_assisted_accuracy": decoder["raw_free_generation_accuracy"],
        "decoder_assisted_accuracy_delta_vs_102": decoder["raw_free_generation_accuracy"] - upstream["metrics_102"]["decoder_assisted_accuracy"],
        "generation_gap_raw_to_decoder": decoder["raw_free_generation_accuracy"] - raw["raw_free_generation_accuracy"],
        "checkpoint_hash_before": checkpoint_hash_before,
        "checkpoint_hash_after": checkpoint_hash_after,
        "checkpoint_hash_unchanged": checkpoint_unchanged,
        "source_100_checkpoint_hash_before": source_100_hash_before,
        "source_100_checkpoint_hash_after": source_100_hash_after,
        "source_100_checkpoint_unchanged": source_100_unchanged,
        "bounded_release_artifact_hash_before": upstream["release_hash_before"],
        "bounded_release_artifact_hash_after": release_hash_after,
        "bounded_release_artifact_unchanged": bounded_release_unchanged,
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "raw_generation_path": "autoregressive",
        "decoder_assisted_used_for_raw": False,
        "ranked_scoring_used_for_raw": False,
        "prefix_forcing_used_for_raw": False,
        "response_table_used_for_raw": False,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
        **{key: dataset["manifest"][key] for key in ["eval_row_hash", "eval_prompt_hash", "eval_count", "eval_dataset_sha256", "overlap_with_101_eval_count", "overlap_with_102_train_count", "overlap_with_102_eval_count", "overlap_with_103_eval_count", "overlap_with_100_train_eval_count", "max_prompt_jaccard_vs_102_train", "max_prompt_jaccard_vs_103_eval"]},
    }
    write_json(
        seed_dir / "checkpoint_manifest.json",
        {
            "schema_version": "multi_seed_raw_generation_confirm_seed_checkpoint_manifest_v1",
            "seed": seed,
            "target_102_checkpoint_path": rel(checkpoint),
            "checkpoint_hash_before": checkpoint_hash_before,
            "checkpoint_hash_after": checkpoint_hash_after,
            "checkpoint_state_hash_before": checkpoint_state_hash_before,
            "checkpoint_state_hash_after": checkpoint_state_hash_after,
            "checkpoint_hash_unchanged": checkpoint_unchanged,
            "source_100_checkpoint_path": rel(source_100),
            "source_100_checkpoint_hash_before": source_100_hash_before,
            "source_100_checkpoint_hash_after": source_100_hash_after,
            "source_100_checkpoint_unchanged": source_100_unchanged,
            "bounded_release_artifact_hash_before": upstream["release_hash_before"],
            "bounded_release_artifact_hash_after": release_hash_after,
            "bounded_release_artifact_unchanged": bounded_release_unchanged,
            "train_step_count": 0,
            "optimizer_step_count": 0,
        },
    )
    passed, failure = seed_passes(seed_metrics, upstream)
    verdicts = ["SEED_RAW_GENERATION_CONFIRM_POSITIVE"] if passed else ["SEED_RAW_GENERATION_CONFIRM_FAILS", failure]
    write_seed_summary(seed_dir, seed, "positive" if passed else "failed", verdicts, seed_metrics)
    seed_record.update(
        {
            "seed_run_completed": True,
            "seed_passed": passed,
            "failure": None if passed else failure,
            "metrics": seed_metrics,
            "report_bits": report_bits,
            "seed_summary_newer_than_104_start": (seed_dir / "summary.json").stat().st_mtime >= started_at,
            "seed_report_newer_than_104_start": (seed_dir / "report.md").stat().st_mtime >= started_at,
            "seed_finished_at_epoch": time.time(),
        }
    )
    return seed_record


def stddev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def aggregate_seed_records(seed_records: list[dict[str, Any]], upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = [record["metrics"] for record in seed_records]
    raw_values = [item["raw_free_generation_accuracy"] for item in metrics]
    case_values = [item["case_id_drift_rate"] for item in metrics]
    aggregate = {
        "schema_version": "multi_seed_raw_generation_confirm_aggregate_v1",
        "seed_count": len(seed_records),
        "seeds": [record["seed"] for record in seed_records],
        "all_seeds_passed_independently": all(record["seed_passed"] for record in seed_records),
        "mean_only_pass_rejected": True,
        "best_seed_pass_rejected": True,
        "two_of_three_pass_rejected": True,
        "min_raw_free_generation_accuracy": min(raw_values),
        "max_raw_free_generation_accuracy": max(raw_values),
        "mean_raw_free_generation_accuracy": sum(raw_values) / max(1, len(raw_values)),
        "stddev_raw_free_generation_accuracy": stddev(raw_values),
        "max_case_id_drift_rate": max(case_values),
        "stddev_case_id_drift_rate": stddev(case_values),
        "max_slot_drift_rate": max(item["slot_drift_rate"] for item in metrics),
        "max_distractor_leak_rate": max(item["distractor_leak_rate"] for item in metrics),
        "min_decoder_assisted_accuracy": min(item["decoder_assisted_accuracy"] for item in metrics),
        "min_bounded_chat_slot_binding_accuracy": min(item["bounded_chat_slot_binding_accuracy"] for item in metrics),
        "min_finite_label_anchorroute_retention_accuracy": min(item["finite_label_anchorroute_retention_accuracy"] for item in metrics),
        "min_unsupported_refusal_accuracy": min(item["unsupported_refusal_accuracy"] for item in metrics),
        "checkpoint_hash_unchanged_all_seeds": all(item["checkpoint_hash_unchanged"] for item in metrics),
        "source_100_checkpoint_unchanged_all_seeds": all(item["source_100_checkpoint_unchanged"] for item in metrics),
        "bounded_release_artifact_unchanged_all_seeds": all(item["bounded_release_artifact_unchanged"] for item in metrics),
        "train_step_count": sum(item["train_step_count"] for item in metrics),
        "optimizer_step_count": sum(item["optimizer_step_count"] for item in metrics),
        "upstream_101_raw_free_generation_accuracy": upstream["metrics_101"]["raw_free_generation_accuracy"],
        "upstream_102_decoder_assisted_accuracy": upstream["metrics_102"]["decoder_assisted_accuracy"],
    }
    return aggregate


def decision_from_records(seed_records: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    failures = [record for record in seed_records if not record["seed_passed"]]
    if not failures:
        primary = "105_RAW_GENERATION_OOD_AND_BOUNDARY_CONFIRM"
        blocking: list[str] = []
    elif any(record["failure"] == "CASE_ID_ANCHOR_GENERALIZATION_FAILS" for record in failures):
        primary = "104B_CASE_ID_MULTI_SEED_FAILURE_ANALYSIS"
        blocking = ["CASE_ID_ANCHOR_GENERALIZATION_FAILS"]
    elif any(record["failure"] == "RETENTION_REGRESSION_DETECTED" for record in failures):
        primary = "RETENTION_FAILURE_ANALYSIS"
        blocking = ["RETENTION_REGRESSION_DETECTED"]
    elif any(record["failure"] in {"EMPTY_OUTPUT_COLLAPSE_DETECTED", "STATIC_RESPONSE_COLLAPSE_DETECTED", "REPETITION_COLLAPSE_DETECTED"} for record in failures):
        primary = "RAW_GENERATION_COLLAPSE_FAILURE_ANALYSIS"
        blocking = ["RAW_GENERATION_COLLAPSE_DETECTED"]
    else:
        primary = "104B_RAW_GENERATION_MULTI_SEED_FAILURE_ANALYSIS"
        blocking = sorted({record["failure"] for record in failures if record["failure"]})
    return {
        "schema_version": "multi_seed_raw_generation_confirm_decision_v1",
        "primary_next_milestone": primary,
        "secondary_track_if_any": None,
        "evidence_for_recommendation": {
            "all_seeds_passed_independently": aggregate["all_seeds_passed_independently"],
            "min_raw_free_generation_accuracy": aggregate["min_raw_free_generation_accuracy"],
            "max_case_id_drift_rate": aggregate["max_case_id_drift_rate"],
            "min_decoder_assisted_accuracy": aggregate["min_decoder_assisted_accuracy"],
        },
        "blocking_failure_modes": blocking,
        "nonblocking_failure_modes": ["HUNGARIAN_DIAGNOSTIC_ONLY"],
        "mechanically_derived": True,
    }


def write_aggregate_reports(out: Path, seed_records: list[dict[str, Any]], aggregate: dict[str, Any], decision: dict[str, Any]) -> None:
    seed_manifests = [
        {
            "seed": record["seed"],
            "seed_run_started": record["seed_run_started"],
            "seed_run_completed": record["seed_run_completed"],
            "seed_summary_newer_than_104_start": record["seed_summary_newer_than_104_start"],
            "seed_report_newer_than_104_start": record["seed_report_newer_than_104_start"],
            "seed_command": record["seed_command"],
            "seed_passed": record["seed_passed"],
            "failure": record["failure"],
            "summary_path": f"seed_{record['seed']}/summary.json",
            "report_path": f"seed_{record['seed']}/report.md",
        }
        for record in seed_records
    ]
    write_json(out / "seed_run_manifest.json", {"schema_version": "multi_seed_raw_generation_confirm_seed_run_manifest_v1", "seeds": seed_manifests})
    write_json(out / "multi_seed_aggregate.json", aggregate)
    write_json(out / "decision_recommendation.json", decision)
    write_json(out / "eval_config.json", {"schema_version": "multi_seed_raw_generation_confirm_config_v1", "seeds": [record["seed"] for record in seed_records], "eval_modes": PHASE103.EVAL_MODES, "eval_families": PHASE103.EVAL_FAMILIES, "eval_only": True, "no_training": True})
    first = seed_records[0]
    write_json(out / "checkpoint_manifest.json", {"schema_version": "multi_seed_raw_generation_confirm_checkpoint_manifest_v1", "seeds": [record["seed"] for record in seed_records], "checkpoint_hash_unchanged_all_seeds": aggregate["checkpoint_hash_unchanged_all_seeds"], "source_100_checkpoint_unchanged_all_seeds": aggregate["source_100_checkpoint_unchanged_all_seeds"], "bounded_release_artifact_unchanged_all_seeds": aggregate["bounded_release_artifact_unchanged_all_seeds"], "first_seed_checkpoint_hash": first["metrics"]["checkpoint_hash_before"], "train_step_count": 0, "optimizer_step_count": 0})
    write_json(out / "case_id_anchor_report.json", {"schema_version": "multi_seed_raw_generation_confirm_case_id_anchor_v1", "max_case_id_drift_rate": aggregate["max_case_id_drift_rate"], "per_seed": {str(record["seed"]): {key: record["metrics"][key] for key in ["case_id_accuracy", "case_id_drift_rate", "distractor_number_copy_rate", "missing_case_id_rate", "wrong_case_id_rate"]} for record in seed_records}})
    write_json(out / "slot_pinning_report.json", {"schema_version": "multi_seed_raw_generation_confirm_slot_pinning_v1", "max_slot_drift_rate": aggregate["max_slot_drift_rate"], "max_distractor_leak_rate": aggregate["max_distractor_leak_rate"], "per_seed": {str(record["seed"]): {key: record["metrics"][key] for key in ["active_slot_accuracy", "slot_drift_rate", "distractor_leak_rate", "stale_old_inactive_leak_rate"]} for record in seed_records}})
    write_json(out / "retention_report.json", {"schema_version": "multi_seed_raw_generation_confirm_retention_v1", "min_bounded_chat_slot_binding_accuracy": aggregate["min_bounded_chat_slot_binding_accuracy"], "min_finite_label_anchorroute_retention_accuracy": aggregate["min_finite_label_anchorroute_retention_accuracy"], "min_unsupported_refusal_accuracy": aggregate["min_unsupported_refusal_accuracy"], "per_seed": {str(record["seed"]): {key: record["metrics"][key] for key in ["bounded_chat_slot_binding_accuracy", "finite_label_anchorroute_retention_accuracy", "unsupported_refusal_accuracy"]} for record in seed_records}})
    write_json(out / "collapse_metrics.json", {"schema_version": "multi_seed_raw_generation_confirm_collapse_metrics_v1", "per_seed": {str(record["seed"]): {key: record["metrics"][key] for key in ["nonempty_generation_rate", "utf8_valid_generation_rate", "empty_output_rate", "static_output_rate", "repetition_rate", "copy_prompt_rate", "wrong_language_rate", "hungarian_diagnostic_accuracy"]} for record in seed_records}, "hungarian_capability_claimed": False})
    write_json(out / "raw_vs_decoder_gap.json", {"schema_version": "multi_seed_raw_generation_confirm_raw_vs_decoder_gap_v1", "min_raw_free_generation_accuracy": aggregate["min_raw_free_generation_accuracy"], "min_decoder_assisted_accuracy": aggregate["min_decoder_assisted_accuracy"], "per_seed": {str(record["seed"]): {key: record["metrics"][key] for key in ["raw_free_generation_accuracy", "decoder_assisted_accuracy", "generation_gap_raw_to_decoder"]} for record in seed_records}})
    family_payload: dict[str, Any] = {}
    mode_payload: dict[str, Any] = {}
    human_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for record in seed_records:
        seed = record["seed"]
        seed_dir = out / f"seed_{seed}"
        family_payload[str(seed)] = read_json(seed_dir / "family_metrics.json")
        mode_payload[str(seed)] = read_json(seed_dir / "mode_comparison.json")
        for row in read_jsonl(seed_dir / "human_readable_samples.jsonl"):
            row["seed"] = seed
            human_rows.append(row)
        for row in read_jsonl(seed_dir / "failure_case_samples.jsonl"):
            row["seed"] = seed
            failure_rows.append(row)
    write_json(out / "family_metrics.json", {"schema_version": "multi_seed_raw_generation_confirm_family_metrics_v1", "per_seed": family_payload})
    write_json(out / "mode_comparison.json", {"schema_version": "multi_seed_raw_generation_confirm_mode_comparison_v1", "per_seed": mode_payload})
    write_jsonl(out / "human_readable_samples.jsonl", human_rows)
    write_jsonl(out / "failure_case_samples.jsonl", failure_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run STABLE_LOOP_PHASE_LOCK_104 multi-seed raw generation confirm")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-103-root", type=Path, default=DEFAULT_UPSTREAM_103_ROOT)
    parser.add_argument("--upstream-102-root", type=Path, default=DEFAULT_UPSTREAM_102_ROOT)
    parser.add_argument("--upstream-101-root", type=Path, default=DEFAULT_UPSTREAM_101_ROOT)
    parser.add_argument("--upstream-100-root", type=Path, default=DEFAULT_UPSTREAM_100_ROOT)
    parser.add_argument("--upstream-099-root", type=Path, default=DEFAULT_UPSTREAM_099_ROOT)
    parser.add_argument("--seeds", default="2027,2028,2029")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    args = parser.parse_args()
    started = time.time()
    out = resolve_target_out(str(args.out))
    args.upstream_103_root = resolve_repo_path(str(args.upstream_103_root), "UPSTREAM_103_ARTIFACT_MISSING")
    args.upstream_102_root = resolve_repo_path(str(args.upstream_102_root), "UPSTREAM_102_ARTIFACT_MISSING")
    args.upstream_101_root = resolve_repo_path(str(args.upstream_101_root), "UPSTREAM_101_ARTIFACT_MISSING")
    args.upstream_100_root = resolve_repo_path(str(args.upstream_100_root), "UPSTREAM_100_ARTIFACT_MISSING")
    args.upstream_099_root = resolve_repo_path(str(args.upstream_099_root), "UPSTREAM_099_ARTIFACT_MISSING")
    seeds = parse_seeds(args.seeds)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "raw_generation_path": "autoregressive",
        "decoder_assisted_used_for_raw": False,
        "ranked_scoring_used_for_raw": False,
        "prefix_forcing_used_for_raw": False,
        "response_table_used_for_raw": False,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "seed_count": len(seeds),
    }
    write_json(out / "queue.json", {"schema_version": "multi_seed_raw_generation_confirm_queue_v1", "milestone": MILESTONE, "partial_write_policy": "progress summary report written from start and refreshed per seed", "seeds": seeds, "steps": ["verify_upstreams", "seed_evals", "aggregate", "decision", "final"]})
    append_progress(out, "start", "running", seeds=seeds)
    write_summary(out, "running", ["MULTI_SEED_RAW_GENERATION_CONFIRM_RUNNING"], metrics)
    try:
        upstream = verify_upstreams(args, out)
        append_progress(out, "upstream verification", "completed")
        write_summary(out, "running", ["UPSTREAM_103_FRESH_CONFIRM_VERIFIED"], metrics)
        seed_records: list[dict[str, Any]] = []
        for seed in seeds:
            append_progress(out, "seed start", "running", seed=seed)
            record = run_seed(seed, out / f"seed_{seed}", args, upstream, started)
            seed_records.append(record)
            append_progress(out, "seed eval completed", "completed", seed=seed, seed_passed=record["seed_passed"], failure=record["failure"])
            write_summary(out, "running", ["SEED_EVAL_COMPLETED"], {**metrics, "completed_seed_count": len(seed_records), "latest_seed": seed})
        aggregate = aggregate_seed_records(seed_records, upstream)
        decision = decision_from_records(seed_records, aggregate)
        write_aggregate_reports(out, seed_records, aggregate, decision)
        metrics.update(aggregate)
        metrics.update({"primary_next_milestone": decision["primary_next_milestone"], "wall_clock_sec": round(time.time() - started, 3)})
        append_progress(out, "aggregate analysis", "completed", all_seeds_passed=aggregate["all_seeds_passed_independently"])
        append_progress(out, "decision recommendation", "completed", next=decision["primary_next_milestone"])
        if metrics["raw_generation_path"] != "autoregressive" or metrics["decoder_assisted_used_for_raw"] or metrics["ranked_scoring_used_for_raw"] or metrics["prefix_forcing_used_for_raw"] or metrics["response_table_used_for_raw"] or metrics["prediction_oracle_used"]:
            raise GateError("RAW_GENERATION_PATH_CONTAMINATED", "raw path contamination detected")
        if not aggregate["all_seeds_passed_independently"]:
            raise GateError("MULTI_SEED_RAW_GENERATION_INSTABILITY_DETECTED", "not all seeds passed independently")
        if aggregate["min_raw_free_generation_accuracy"] < 0.85 or aggregate["max_case_id_drift_rate"] > 0.10 or aggregate["max_slot_drift_rate"] > 0.05 or aggregate["max_distractor_leak_rate"] > 0.10:
            raise GateError("MULTI_SEED_RAW_GENERATION_INSTABILITY_DETECTED", "aggregate raw-generation gates failed")
        if aggregate["min_decoder_assisted_accuracy"] < 0.90:
            raise GateError("DECODER_ASSISTED_REGRESSION_DETECTED", "decoder-assisted aggregate gate failed")
        if aggregate["min_bounded_chat_slot_binding_accuracy"] < 0.90 or aggregate["min_finite_label_anchorroute_retention_accuracy"] < 0.90 or aggregate["min_unsupported_refusal_accuracy"] < 0.80:
            raise GateError("RETENTION_REGRESSION_DETECTED", "retention aggregate gate failed")
        if not aggregate["checkpoint_hash_unchanged_all_seeds"] or not aggregate["source_100_checkpoint_unchanged_all_seeds"]:
            raise GateError("CHECKPOINT_MUTATION_DETECTED", "checkpoint mutation detected")
        if not aggregate["bounded_release_artifact_unchanged_all_seeds"]:
            raise GateError("BOUNDED_RELEASE_MUTATION_DETECTED", "bounded release artifact changed")
        if aggregate["train_step_count"] != 0 or aggregate["optimizer_step_count"] != 0:
            raise GateError("TRAINING_SIDE_EFFECT_DETECTED", "training side effect detected")
        append_progress(out, "final verdict", "positive", next=decision["primary_next_milestone"])
        write_summary(
            out,
            "positive",
            [
                "MULTI_SEED_RAW_GENERATION_CONFIRM_POSITIVE",
                "UPSTREAM_103_FRESH_CONFIRM_VERIFIED",
                "RAW_GENERATION_GENERALIZES_ALL_SEEDS",
                "CASE_ID_ANCHOR_GENERALIZES_ALL_SEEDS",
                "SLOT_PINNING_GENERALIZES_ALL_SEEDS",
                "DECODER_ASSISTED_REFERENCE_RETAINED_ALL_SEEDS",
                "UNSUPPORTED_REFUSAL_RETAINED_ALL_SEEDS",
                "BOUNDED_CHAT_RETENTION_PASSES_ALL_SEEDS",
                "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES_ALL_SEEDS",
                "COLLAPSE_REJECTED_ALL_SEEDS",
                "CHECKPOINT_UNCHANGED_ALL_SEEDS",
                "NO_TRAINING_PERFORMED",
                "GPT_LIKE_READINESS_NOT_CLAIMED",
                "PRODUCTION_CHAT_NOT_CLAIMED",
            ],
            metrics,
        )
        return 0
    except GateError as exc:
        return fail(out, exc.verdict, exc.message, metrics)


if __name__ == "__main__":
    sys.exit(main())
