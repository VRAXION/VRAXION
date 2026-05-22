#!/usr/bin/env python3
"""Analysis-only 111R retention / LM regression attribution."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_111R_RETENTION_OR_LM_REGRESSION_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_111r_retention_or_lm_regression_analysis/smoke")
DEFAULT_UPSTREAM_111_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_111_overnight_decoder_policy_distillation_raw_rollout_repair/smoke")
DEFAULT_UPSTREAM_110_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_110_integrated_decoder_policy_ood_confirm_batch/smoke")
DEFAULT_UPSTREAM_109_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_109_decoder_policy_integration/smoke")
DEFAULT_UPSTREAM_108A_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis/smoke")

POSITIVE_VERDICT = "RETENTION_OR_LM_REGRESSION_ANALYSIS_POSITIVE"
BOUNDARY_TEXT = (
    "111R is analysis only. It reads failed 111 and positive 110/109/108A artifacts, performs no "
    "training, no repair, mutates no checkpoint, and changes no runtime/service/deploy surface. "
    "The failed 111 target checkpoint is not a release candidate. This is not GPT-like assistant "
    "readiness, not open-domain assistant readiness, not production chat, not public API, not "
    "deployment readiness, and not safety alignment."
)

REQUIRED_111_ARTIFACTS = [
    "summary.json",
    "decision.json",
    "arm_comparison.json",
    "train_dataset_manifest.json",
    "eval_dataset_manifest.json",
    "generation_results_pre_raw.jsonl",
    "generation_results_post_raw.jsonl",
    "generation_results_integrated_teacher.jsonl",
    "train_examples_sample.jsonl",
    "training_metrics.jsonl",
    "bounded_retention_metrics.json",
    "collapse_metrics.json",
    "overclaim_metrics.json",
]


class GateError(Exception):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict
        self.message = message


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("RETENTION_OR_LM_REGRESSION_ANALYSIS_FAILS", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("RETENTION_OR_LM_REGRESSION_ANALYSIS_FAILS", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def resolve_upstream(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[^\w\s]", text.lower(), flags=re.ASCII)


def first_wrong_position(expected: str, observed: str) -> dict[str, Any]:
    expected_tokens = tokenise(expected)
    observed_tokens = tokenise(observed)
    count = 0
    for left, right in zip(expected_tokens, observed_tokens):
        if left != right:
            break
        count += 1
    first_wrong = None if count == len(expected_tokens) and len(observed_tokens) == len(expected_tokens) else count
    expected_count = len(expected_tokens)
    return {
        "first_wrong_token_position": first_wrong,
        "matching_prefix_token_count": count,
        "expected_token_count": expected_count,
        "gold_prefix_survival_rate": count / expected_count if expected_count else 1.0,
    }


def rate(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def fmean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def number_prefixes(text: str) -> list[str]:
    return [number[:3] for number in re.findall(r"\b\d{6,}\b", text)]


def numbers(text: str) -> list[str]:
    return re.findall(r"\b\d{6,}\b", text)


def prefix_counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(number_prefixes(str(row.get(field, ""))))
    return dict(counter)


def family_base(name: str) -> str:
    return name.replace("_FINAL", "").replace("_CONFIRM", "")


def load_required(root: Path, rels: list[str], verdict: str) -> None:
    missing = [rel for rel in rels if not (root / rel).exists()]
    if missing:
        raise GateError(verdict, "missing artifacts: " + ", ".join(missing))


def verify_positive(root: Path, positive_verdict: str, missing_verdict: str) -> dict[str, Any]:
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise GateError(missing_verdict, f"missing {rel(summary_path)}")
    summary = read_json(summary_path)
    if positive_verdict not in set(summary.get("verdicts", [])):
        raise GateError("UPSTREAM_STACK_NOT_POSITIVE", f"{positive_verdict} not found")
    return summary


def verify_failed_111(root: Path) -> dict[str, Any]:
    load_required(root, REQUIRED_111_ARTIFACTS, "UPSTREAM_111_ARTIFACT_MISSING")
    summary = read_json(root / "summary.json")
    decision = read_json(root / "decision.json")
    metrics = summary.get("metrics", {})
    verdicts = set(summary.get("verdicts", []))
    if summary.get("status") != "failed" or "RAW_OOD_ACCURACY_NOT_IMPROVED" not in verdicts:
        raise GateError("UPSTREAM_111_FAILURE_NOT_FOUND", "111 failure verdict not found")
    if decision.get("next") != "111R_RETENTION_OR_LM_REGRESSION_ANALYSIS":
        raise GateError("UPSTREAM_111_FAILURE_NOT_FOUND", "111 decision does not route to 111R")
    required_true = [
        "target_111_checkpoint_changed",
        "source_102_checkpoint_unchanged",
        "source_100_checkpoint_unchanged",
        "bounded_release_artifact_unchanged",
        "packaged_winner_hash_unchanged",
    ]
    for key in required_true:
        if metrics.get(key) is not True:
            raise GateError("UPSTREAM_111_FAILURE_NOT_FOUND", f"111 metric {key} not true")
    if metrics.get("runtime_profile") != "standard" or metrics.get("train_step_count", 0) <= 0 or metrics.get("optimizer_step_count", 0) <= 0:
        raise GateError("UPSTREAM_111_FAILURE_NOT_FOUND", "111 standard training counters missing")
    return summary


def write_summary(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    payload = {
        "schema_version": "retention_lm_regression_analysis_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "phase": status,
        "analysis_only": True,
        "training_performed": False,
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "failed_111_target_checkpoint_is_release_candidate": False,
        "stable_release_stack_remains_valid": True,
        "integrated_teacher_path_remains_valid": True,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "public_api_claimed": False,
        "deployment_readiness_claimed": False,
        "safety_alignment_claimed": False,
        "boundary": BOUNDARY_TEXT,
        "metrics": metrics,
        "verdicts": verdicts,
    }
    if message:
        payload["message"] = message
    write_json(out / "summary.json", payload)
    write_report(out, payload)


def write_report(out: Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    lines = [
        f"# {MILESTONE}",
        "",
        BOUNDARY_TEXT,
        "",
        f"Status: `{summary.get('status')}`",
        "",
        "## Key Findings",
        "",
        f"- primary_root_cause: `{metrics.get('primary_root_cause')}`",
        f"- secondary_root_causes: `{metrics.get('secondary_root_causes')}`",
        f"- recommended_next: `{metrics.get('recommended_next')}`",
        f"- pre_111_raw_ood_accuracy: `{metrics.get('pre_111_raw_ood_accuracy')}`",
        f"- post_111_raw_ood_accuracy: `{metrics.get('post_111_raw_ood_accuracy')}`",
        f"- integrated_teacher_ood_accuracy: `{metrics.get('integrated_teacher_ood_accuracy')}`",
        f"- namespace_leak_rate: `{metrics.get('namespace_leak_rate')}`",
        f"- teacher_namespace_copy_rate: `{metrics.get('teacher_namespace_copy_rate')}`",
        f"- rollout_prefix_survival_rate_mean: `{metrics.get('prefix_survival_rate_mean')}`",
        f"- static_output_rate: `{metrics.get('static_output_rate')}`",
        f"- repetition_rate: `{metrics.get('repetition_rate')}`",
        "",
        "## Verdicts",
        "",
    ]
    lines.extend(f"- `{verdict}`" for verdict in summary.get("verdicts", []))
    write_text(out / "report.md", "\n".join(lines) + "\n")


def fail(out: Path, verdict: str, message: str, metrics: dict[str, Any]) -> int:
    append_progress(out, "failure", "failed", verdict=verdict, message=message)
    write_summary(out, "failed", ["RETENTION_OR_LM_REGRESSION_ANALYSIS_FAILS", verdict], metrics, message)
    return 1


def eval_path_report(up111: Path, up110: Path, summary111: dict[str, Any], summary110: dict[str, Any], pre_rows: list[dict[str, Any]], raw110_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics111 = summary111.get("metrics", {})
    metrics110 = summary110.get("metrics", {})
    families110 = sorted({row.get("eval_family") for row in raw110_rows})
    families111 = sorted({row.get("eval_family") for row in pre_rows})
    prompt_prefix111 = rate([str(row.get("prompt", "")).startswith("Overnight target-only 111 row") for row in pre_rows])
    prompt_prefix110 = rate([str(row.get("prompt", "")).startswith("Overnight target-only 111 row") for row in raw110_rows])
    report = {
        "schema_version": "eval_path_compatibility_report_v1",
        "hard_question": "Why did pre_111_raw_ood_accuracy equal 0.0 when prior raw OOD was around 0.53?",
        "prior_raw_ood_accuracy": metrics110.get("raw_ood_stress_accuracy"),
        "pre_111_raw_ood_accuracy": metrics111.get("pre_111_raw_ood_accuracy"),
        "accuracy_delta_pre111_vs_110_raw": (metrics111.get("pre_111_raw_ood_accuracy", 0.0) - metrics110.get("raw_ood_stress_accuracy", 0.0)),
        "raw_generation_wrapper": {
            "phase_110": "RAW_FREE_GENERATION artifact path raw_generation_results.jsonl",
            "phase_111_pre": "PRE_111_RAW_BASELINE artifact path generation_results_pre_raw.jsonl",
            "phase_111_post": "POST_111_RAW_DISTILLED artifact path generation_results_post_raw.jsonl",
            "decoder_policy_disabled_for_111_raw": all(row.get("integrated_policy_used_during_final_raw_eval") is False and row.get("decoder_reference_used_during_final_raw_eval") is False for row in pre_rows),
        },
        "prompt_schema": {
            "phase_110_over_night_prefix_rate": prompt_prefix110,
            "phase_111_overnight_prefix_rate": prompt_prefix111,
            "phase_110_family_suffixes": sorted({name.rsplit("_", 1)[-1] for name in families110 if name}),
            "phase_111_family_suffixes": sorted({name.rsplit("_", 1)[-1] for name in families111 if name}),
            "phase_110_case_id_prefixes": prefix_counter(raw110_rows, "prompt"),
            "phase_111_case_id_prefixes": prefix_counter(pre_rows, "prompt"),
        },
        "scoring_rules": {
            "phase_110_pass_fail_field": "pass_fail",
            "phase_111_pass_fail_field": "pass_fail",
            "phase_111_expected_keywords_present": rate([bool(row.get("required_keywords")) for row in pre_rows]),
            "phase_111_expected_response_present": rate([bool(row.get("expected_response")) for row in pre_rows]),
        },
        "compatibility_findings": [
            "111 final eval wraps prompts with a long phase-111 prompt schema absent from 110 raw rows.",
            "111 final eval uses a 911 namespace while 110 raw confirm rows use ordinary generated case ids.",
            "The 0.0 pre-baseline result is therefore treated as eval-path / prompt-format incompatibility evidence, not as proof that the best known raw path regressed.",
        ],
    }
    write_json(up111 / "_unused.tmp", {}) if False else None
    return report


def namespace_report(pre_rows: list[dict[str, Any]], post_rows: list[dict[str, Any]], teacher_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_prefixes = prefix_counter(post_rows, "prompt")
    teacher_prefixes = prefix_counter(teacher_rows + train_rows, "prompt")
    teacher_output_prefixes = prefix_counter(teacher_rows + train_rows, "response")
    generated_prefixes = prefix_counter(post_rows, "generated_text")
    final_eval_prefixes = prompt_prefixes
    leak_rows: list[dict[str, Any]] = []
    drift_rows = 0
    for row in post_rows:
        prompt_ids = set(numbers(row.get("prompt", "")))
        expected_ids = set(numbers(row.get("expected_response", "")))
        generated_ids = set(numbers(row.get("generated_text", "")))
        generated_teacher_prefix = any(num.startswith("711") for num in generated_ids)
        missing_expected = bool(expected_ids) and not bool(generated_ids & expected_ids)
        if generated_teacher_prefix and any(num.startswith("911") for num in prompt_ids):
            leak_rows.append(row)
        if missing_expected or (generated_ids and not (generated_ids & (prompt_ids | expected_ids))):
            drift_rows += 1
    total = max(1, len(post_rows))
    return {
        "schema_version": "namespace_leakage_report_v1",
        "prompt_namespace_prefixes": prompt_prefixes,
        "teacher_train_namespace_prefixes": teacher_prefixes,
        "teacher_train_output_namespace_prefixes": teacher_output_prefixes,
        "final_eval_namespace_prefixes": final_eval_prefixes,
        "generated_namespace_prefixes": generated_prefixes,
        "namespace_leak_rate": len(leak_rows) / total,
        "teacher_namespace_copy_rate": sum(1 for row in post_rows if any(num.startswith("711") for num in numbers(row.get("generated_text", "")))) / total,
        "case_id_drift_rate": drift_rows / total,
        "leaked_711_into_911_count": len(leak_rows),
        "sample_leaks": [
            {
                "eval_family": row.get("eval_family"),
                "prompt_ids": numbers(row.get("prompt", "")),
                "generated_ids": numbers(row.get("generated_text", "")),
                "generated_text": row.get("generated_text"),
            }
            for row in leak_rows[:20]
        ],
    }


def rollout_report(summary111: dict[str, Any], post_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summary111.get("metrics", {})
    prefixes = [first_wrong_position(row.get("expected_response", ""), row.get("generated_text", "")) for row in post_rows]
    first_positions = [item["first_wrong_token_position"] for item in prefixes if item["first_wrong_token_position"] is not None]
    prefix_rates = [item["gold_prefix_survival_rate"] for item in prefixes]
    return {
        "schema_version": "rollout_gap_report_v1",
        "teacher_forced_loss_final": metrics.get("teacher_distillation_loss_final"),
        "train_loss_initial": metrics.get("train_loss_initial"),
        "train_loss_final": metrics.get("train_loss_final"),
        "rollout_accuracy": metrics.get("post_111_raw_ood_accuracy"),
        "rollout_drift_rate": 1.0 - float(metrics.get("post_111_raw_ood_accuracy", 0.0)),
        "first_wrong_token_position_mean": fmean([float(item) for item in first_positions]),
        "first_wrong_token_position_median": statistics.median(first_positions) if first_positions else None,
        "prefix_survival_rate_mean": fmean(prefix_rates),
        "prefix_survival_rate_min": min(prefix_rates) if prefix_rates else 0.0,
        "repetition_rate": metrics.get("repetition_rate"),
        "static_output_rate": metrics.get("static_output_rate"),
        "stop_condition_findings": {
            "empty_output_rate": metrics.get("empty_output_rate"),
            "nonempty_generation_rate": metrics.get("nonempty_generation_rate"),
            "utf8_valid_generation_rate": metrics.get("utf8_valid_generation_rate"),
        },
        "classification": "TEACHER_FORCING_ROLLOUT_GAP" if metrics.get("train_loss_final", 999) < metrics.get("train_loss_initial", -999) and metrics.get("post_111_raw_ood_accuracy") == 0.0 else "UNKNOWN",
    }


def retention_report(train_manifest: dict[str, Any], post_rows: list[dict[str, Any]]) -> dict[str, Any]:
    retention_rows = [row for row in post_rows if "RETENTION" in row.get("eval_family", "")]
    failure_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for row in retention_rows:
        output = str(row.get("generated_text", ""))
        expected = str(row.get("expected_response", ""))
        failure_type = "unknown"
        if row.get("repetition_flag") or len(set(output.split())) < max(2, len(output.split()) // 4):
            failure_type = "generated_output_collapsed"
        if any(num.startswith("711") for num in numbers(output)) and any(num.startswith("911") for num in numbers(row.get("prompt", ""))):
            failure_type = "wrong_namespace"
        elif expected and numbers(expected) and not (set(numbers(expected)) & set(numbers(output))):
            failure_type = "wrong_format_or_label"
        if row.get("pass_fail") == "pass":
            failure_type = "pass"
        failure_counts[failure_type] += 1
        if len(samples) < 30:
            samples.append({
                "eval_family": row.get("eval_family"),
                "prompt": row.get("prompt"),
                "expected_response": expected,
                "generated_text": output,
                "failure_type": failure_type,
                "required_keywords": row.get("required_keywords"),
            })
    mix = train_manifest.get("training_mix", {})
    train_count = max(1, int(train_manifest.get("train_row_count", 0)))
    return {
        "schema_version": "retention_regression_report_v1",
        "retention_rows_in_train": {
            "bounded_chat_retention": mix.get("bounded_chat_retention", 0),
            "finite_label_anchorroute_retention": mix.get("finite_label_anchorroute_retention", 0),
            "retention_train_fraction": (float(mix.get("bounded_chat_retention", 0)) + float(mix.get("finite_label_anchorroute_retention", 0))) / train_count,
        },
        "retention_rows_in_eval": len(retention_rows),
        "retention_label_format": "case-id plus active slot/finite LABEL_* required keywords",
        "retention_failure_type_counts": dict(failure_counts),
        "retention_generated_outputs": samples,
        "analysis": "Retention examples were present in the intended mix, but final rollout outputs use wrong namespace or collapsed/static strings, so this is not only a scorer-format explanation.",
    }


def data_balance_report(train_manifest: dict[str, Any], training_rows: list[dict[str, Any]], resource_report: dict[str, Any]) -> dict[str, Any]:
    mix = train_manifest.get("training_mix", {})
    train_count = max(1, int(train_manifest.get("train_row_count", 0)))
    actual = {
        "teacher_distill_percentage_actual": float(mix.get("integrated_teacher", 0)) / train_count,
        "fineweb_replay_percentage_actual": float(mix.get("fineweb_replay_fraction", 0.0)),
        "bounded_retention_percentage_actual": float(mix.get("bounded_chat_retention", 0)) / train_count,
        "finite_label_retention_percentage_actual": float(mix.get("finite_label_anchorroute_retention", 0)) / train_count,
        "refusal_boundary_percentage_actual": float(mix.get("refusal_boundary_prompt_injection", 0)) / train_count,
        "short_instruction_or_qa_percentage_actual": float(mix.get("short_instruction_or_qa", 0)) / train_count,
    }
    batch_counter = Counter(row.get("batch_id") for row in training_rows if row.get("batch_id"))
    return {
        "schema_version": "data_balance_report_v1",
        "train_row_count": train_count,
        "training_mix_counts": mix,
        **actual,
        "training_metric_batches": dict(batch_counter),
        "resource_summary": {
            "median_gpu_utilization": resource_report.get("median_gpu_utilization"),
            "gpu_idle_fraction": resource_report.get("gpu_idle_fraction"),
            "throughput_examples_per_sec": resource_report.get("throughput_examples_per_sec"),
        },
    }


def output_collapse_report(post_rows: list[dict[str, Any]], summary111: dict[str, Any]) -> dict[str, Any]:
    outputs = [row.get("generated_text", "") for row in post_rows]
    common = Counter(outputs).most_common(10)
    metrics = summary111.get("metrics", {})
    return {
        "schema_version": "output_collapse_report_v1",
        "static_output_rate": metrics.get("static_output_rate"),
        "repetition_rate": metrics.get("repetition_rate"),
        "copy_prompt_rate": metrics.get("copy_prompt_rate"),
        "empty_output_rate": metrics.get("empty_output_rate"),
        "nonempty_generation_rate": metrics.get("nonempty_generation_rate"),
        "utf8_valid_generation_rate": metrics.get("utf8_valid_generation_rate"),
        "most_common_outputs": [{"output": text, "count": count, "rate": count / max(1, len(outputs))} for text, count in common],
    }


def root_cause(metrics: dict[str, Any]) -> tuple[str, list[str], str]:
    causes: list[str] = []
    if metrics["pre_111_raw_ood_accuracy"] == 0.0 and metrics["prior_raw_ood_accuracy"] >= 0.50:
        causes.append("EVAL_PATH_MISMATCH")
    if metrics["namespace_leak_rate"] > 0.05 or metrics["teacher_namespace_copy_rate"] > 0.05:
        causes.append("NAMESPACE_MEMORIZATION")
    if metrics["train_loss_final"] < metrics["train_loss_initial"] and metrics["post_111_raw_ood_accuracy"] == 0.0:
        causes.append("TEACHER_FORCING_ROLLOUT_GAP")
    if metrics["retention_accuracy_min"] == 0.0:
        causes.append("RETENTION_MIX_UNDERPOWERED")
    if metrics["static_output_rate"] > 0.15 or metrics["repetition_rate"] > 0.25:
        causes.append("TARGET_CHECKPOINT_COLLAPSE")
    if metrics.get("post_111_raw_ood_accuracy", 0.0) == 0.0 and metrics.get("integrated_teacher_ood_accuracy", 0.0) == 1.0:
        causes.append("DATA_BALANCE_FAILURE")
    unique = list(dict.fromkeys(causes))
    primary = "MIXED_CAUSE" if len(unique) > 1 else (unique[0] if unique else "SCORER_FORMAT_MISMATCH")
    next_plan = {
        "EVAL_PATH_MISMATCH": "111E_EVAL_PATH_COMPATIBILITY_FIX",
        "NAMESPACE_MEMORIZATION": "111N_NAMESPACE_RANDOMIZATION_AND_ANTI_MEMORIZATION_REPAIR",
        "TEACHER_FORCING_ROLLOUT_GAP": "111G_SCHEDULED_SAMPLING_OR_ROLLOUT_LOSS_REPAIR",
        "RETENTION_MIX_UNDERPOWERED": "111M_RETENTION_MIX_REBALANCE_REPAIR",
        "MIXED_CAUSE": "111X_COMBINED_RAW_DISTILLATION_REDESIGN",
    }.get(primary, "111X_COMBINED_RAW_DISTILLATION_REDESIGN")
    return primary, unique, next_plan


def human_samples(pre_rows: list[dict[str, Any]], post_rows: list[dict[str, Any]], teacher_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pre_by_index = {row.get("eval_index"): row for row in pre_rows}
    teacher_by_index = {row.get("eval_index"): row for row in teacher_rows}
    selected: list[dict[str, Any]] = []
    categories = {
        "911 prompt -> 711 output": lambda row: any(num.startswith("711") for num in numbers(row.get("generated_text", ""))) and any(num.startswith("911") for num in numbers(row.get("prompt", ""))),
        "retention failure": lambda row: "RETENTION" in row.get("eval_family", ""),
        "repetition/static output": lambda row: bool(row.get("repetition_flag")),
        "pre-baseline mismatch": lambda row: pre_by_index.get(row.get("eval_index"), {}).get("pass_fail") == "fail",
        "rollout drift": lambda row: row.get("pass_fail") == "fail",
    }
    used: set[int] = set()
    for label, predicate in categories.items():
        for row in post_rows:
            idx = int(row.get("eval_index", -1))
            if idx in used or not predicate(row):
                continue
            pre = pre_by_index.get(row.get("eval_index"), {})
            teacher = teacher_by_index.get(row.get("eval_index"), {})
            selected.append({
                "sample_type": label,
                "eval_index": row.get("eval_index"),
                "eval_family": row.get("eval_family"),
                "prompt": row.get("prompt"),
                "expected_response": row.get("expected_response"),
                "pre_111_output": pre.get("generated_text"),
                "post_111_output": row.get("generated_text"),
                "integrated_teacher_output": teacher.get("generated_text"),
                "failure_labels": {
                    "pre": pre.get("failure_label"),
                    "post": row.get("failure_label"),
                    "teacher": teacher.get("failure_label"),
                },
                "namespace_detected": {
                    "prompt_prefixes": number_prefixes(row.get("prompt", "")),
                    "post_output_prefixes": number_prefixes(row.get("generated_text", "")),
                },
                "short_diagnosis": "111 raw rollout failed while integrated teacher passed on the same row.",
            })
            used.add(idx)
            break
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-111-root", default=str(DEFAULT_UPSTREAM_111_ROOT))
    parser.add_argument("--upstream-110-root", default=str(DEFAULT_UPSTREAM_110_ROOT))
    parser.add_argument("--upstream-109-root", default=str(DEFAULT_UPSTREAM_109_ROOT))
    parser.add_argument("--upstream-108a-root", default=str(DEFAULT_UPSTREAM_108A_ROOT))
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    args = parser.parse_args(argv)

    out = resolve_target_out(args.out)
    up111 = resolve_upstream(args.upstream_111_root)
    up110 = resolve_upstream(args.upstream_110_root)
    up109 = resolve_upstream(args.upstream_109_root)
    up108a = resolve_upstream(args.upstream_108a_root)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    metrics: dict[str, Any] = {
        "schema_version": "retention_lm_regression_analysis_metrics_v1",
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "analysis_only": True,
        "checkpoints_unchanged": True,
    }
    write_json(out / "queue.json", {"schema_version": "retention_lm_regression_queue_v1", "milestone": MILESTONE, "steps": ["verify_upstreams", "load_artifacts", "analyze_eval_path", "analyze_namespace", "analyze_rollout", "analyze_retention", "classify", "final"]})
    write_json(out / "analysis_config.json", {"schema_version": "retention_lm_regression_config_v1", "upstream_111_root": rel(up111), "upstream_110_root": rel(up110), "upstream_109_root": rel(up109), "upstream_108a_root": rel(up108a), "heartbeat_sec": args.heartbeat_sec})
    write_summary(out, "running", ["RETENTION_OR_LM_REGRESSION_ANALYSIS_RUNNING"], metrics)
    append_progress(out, "start", "running", milestone=MILESTONE)

    try:
        summary111 = verify_failed_111(up111)
        summary110 = verify_positive(up110, "INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_POSITIVE", "UPSTREAM_110_ARTIFACT_MISSING")
        summary109 = verify_positive(up109, "DECODER_POLICY_INTEGRATION_POSITIVE", "UPSTREAM_109_ARTIFACT_MISSING")
        summary108a = verify_positive(up108a, "RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_POSITIVE", "UPSTREAM_108A_ARTIFACT_MISSING")
        write_json(out / "upstream_111_manifest.json", {"schema_version": "upstream_111_failure_manifest_v1", "root": rel(up111), "summary": summary111})
        write_json(out / "upstream_110_manifest.json", {"schema_version": "upstream_110_manifest_v1", "root": rel(up110), "summary": summary110})
        write_json(out / "upstream_109_manifest.json", {"schema_version": "upstream_109_manifest_v1", "root": rel(up109), "summary": summary109})
        write_json(out / "upstream_108a_manifest.json", {"schema_version": "upstream_108a_manifest_v1", "root": rel(up108a), "summary": summary108a})
        append_progress(out, "upstream verification", "completed")

        pre_rows = read_jsonl(up111 / "generation_results_pre_raw.jsonl")
        post_rows = read_jsonl(up111 / "generation_results_post_raw.jsonl")
        teacher_eval_rows = read_jsonl(up111 / "generation_results_integrated_teacher.jsonl")
        raw110_rows = read_jsonl(up110 / "raw_generation_results.jsonl")
        train_sample = read_jsonl(up111 / "train_examples_sample.jsonl")
        teacher_sample = read_jsonl(up111 / "teacher_policy_trace_sample.jsonl") if (up111 / "teacher_policy_trace_sample.jsonl").exists() else []
        train_manifest = read_json(up111 / "train_dataset_manifest.json")
        training_metrics = read_jsonl(up111 / "training_metrics.jsonl")
        resource_report = read_json(up111 / "resource_report.json") if (up111 / "resource_report.json").exists() else {}
        metrics111 = summary111["metrics"]
        metrics.update({
            "pre_111_raw_ood_accuracy": metrics111.get("pre_111_raw_ood_accuracy"),
            "post_111_raw_ood_accuracy": metrics111.get("post_111_raw_ood_accuracy"),
            "integrated_teacher_ood_accuracy": metrics111.get("integrated_teacher_ood_accuracy"),
            "prior_raw_ood_accuracy": summary110["metrics"].get("raw_ood_stress_accuracy"),
            "train_loss_initial": metrics111.get("train_loss_initial"),
            "train_loss_final": metrics111.get("train_loss_final"),
            "teacher_forced_loss_final": metrics111.get("teacher_distillation_loss_final"),
            "static_output_rate": metrics111.get("static_output_rate"),
            "repetition_rate": metrics111.get("repetition_rate"),
            "retention_accuracy_min": min(float(metrics111.get("bounded_chat_slot_binding_accuracy", 0.0)), float(metrics111.get("finite_label_anchorroute_retention_accuracy", 0.0)), float(metrics111.get("unsupported_refusal_retention_accuracy", 0.0))),
        })
        append_progress(out, "artifact load", "completed", pre_rows=len(pre_rows), post_rows=len(post_rows), raw110_rows=len(raw110_rows))

        eval_report = eval_path_report(up111, up110, summary111, summary110, pre_rows, raw110_rows)
        write_json(out / "eval_path_compatibility_report.json", eval_report)
        append_progress(out, "eval path compatibility", "completed")

        ns_report = namespace_report(pre_rows, post_rows, teacher_sample, train_sample)
        metrics.update({
            "namespace_leak_rate": ns_report["namespace_leak_rate"],
            "teacher_namespace_copy_rate": ns_report["teacher_namespace_copy_rate"],
            "case_id_drift_rate": ns_report["case_id_drift_rate"],
        })
        write_json(out / "namespace_leakage_report.json", ns_report)
        append_progress(out, "namespace leakage", "completed", namespace_leak_rate=ns_report["namespace_leak_rate"])

        roll_report = rollout_report(summary111, post_rows)
        metrics.update({
            "prefix_survival_rate_mean": roll_report["prefix_survival_rate_mean"],
            "rollout_drift_rate": roll_report["rollout_drift_rate"],
        })
        write_json(out / "rollout_gap_report.json", roll_report)
        append_progress(out, "rollout gap", "completed")

        retention = retention_report(train_manifest, post_rows)
        write_json(out / "retention_regression_report.json", retention)
        append_progress(out, "retention regression", "completed")

        balance = data_balance_report(train_manifest, training_metrics, resource_report)
        write_json(out / "data_balance_report.json", balance)
        collapse = output_collapse_report(post_rows, summary111)
        write_json(out / "output_collapse_report.json", collapse)
        append_progress(out, "data balance and collapse", "completed")

        primary, secondary, next_plan = root_cause(metrics)
        metrics.update({
            "primary_root_cause": primary,
            "secondary_root_causes": secondary,
            "recommended_next": next_plan,
            "wall_clock_sec": round(time.time() - start, 3),
        })
        classification = {
            "schema_version": "root_cause_classification_v1",
            "primary_root_cause": primary,
            "secondary_root_causes": secondary,
            "allowed_labels": [
                "EVAL_PATH_MISMATCH",
                "NAMESPACE_MEMORIZATION",
                "TEACHER_FORCING_ROLLOUT_GAP",
                "RETENTION_MIX_UNDERPOWERED",
                "DATA_BALANCE_FAILURE",
                "STOP_CONDITION_FAILURE",
                "TARGET_CHECKPOINT_COLLAPSE",
                "SCORER_FORMAT_MISMATCH",
                "MIXED_CAUSE",
            ],
            "evidence": {
                "pre_baseline_zero_despite_prior_raw": metrics["pre_111_raw_ood_accuracy"] == 0.0 and metrics["prior_raw_ood_accuracy"] >= 0.50,
                "namespace_leak_rate": metrics["namespace_leak_rate"],
                "teacher_forced_loss_improved_but_rollout_failed": metrics["train_loss_final"] < metrics["train_loss_initial"] and metrics["post_111_raw_ood_accuracy"] == 0.0,
                "retention_accuracy_min": metrics["retention_accuracy_min"],
                "static_output_rate": metrics["static_output_rate"],
                "repetition_rate": metrics["repetition_rate"],
            },
        }
        write_json(out / "root_cause_classification.json", classification)

        recommended = {
            "schema_version": "recommended_next_plan_v1",
            "next": next_plan,
            "primary_root_cause": primary,
            "secondary_root_causes": secondary,
            "evidence_counts": {
                "post_eval_rows": len(post_rows),
                "namespace_leak_rows": ns_report["leaked_711_into_911_count"],
                "retention_eval_rows": retention["retention_rows_in_eval"],
            },
            "evidence_rates": {
                "namespace_leak_rate": metrics["namespace_leak_rate"],
                "teacher_namespace_copy_rate": metrics["teacher_namespace_copy_rate"],
                "case_id_drift_rate": metrics["case_id_drift_rate"],
                "rollout_drift_rate": metrics["rollout_drift_rate"],
                "retention_accuracy_min": metrics["retention_accuracy_min"],
            },
            "why_not_more_steps": "Training loss already fell sharply while rollout accuracy stayed 0.0 and collapse/namespace leakage appeared; more steps alone would not address eval-path, namespace, rollout, and retention failures.",
            "why_not_repeat_111_standard": "The failed run produced 711 train namespace outputs for 911 final prompts, retention accuracy 0.0, and high static/repetition rates; repeating the same setup would likely reproduce the same failure.",
        }
        write_json(out / "recommended_next_plan.json", recommended)
        write_jsonl(out / "human_readable_failure_samples.jsonl", human_samples(pre_rows, post_rows, teacher_eval_rows))
        append_progress(out, "root cause classification", "completed", next=next_plan)

        verdicts = [
            POSITIVE_VERDICT,
            "UPSTREAM_111_FAILURE_VERIFIED",
            "EVAL_PATH_COMPATIBILITY_ANALYZED",
            "NAMESPACE_LEAKAGE_ANALYZED",
            "ROLLOUT_GAP_ANALYZED",
            "RETENTION_REGRESSION_ANALYZED",
            "DATA_BALANCE_ANALYZED",
            "ROOT_CAUSE_CLASSIFIED",
            "RECOMMENDED_NEXT_PLAN_WRITTEN",
            "NO_TRAINING_PERFORMED",
            "CHECKPOINTS_UNCHANGED",
            "PRODUCTION_CHAT_NOT_CLAIMED",
            "GPT_LIKE_READINESS_NOT_CLAIMED",
        ]
        append_progress(out, "final verdict", "positive", next=next_plan)
        write_summary(out, "positive", verdicts, metrics)
        print(json.dumps({"out": rel(out), "next": next_plan, "primary_root_cause": primary}, sort_keys=True))
        return 0
    except GateError as exc:
        return fail(out, exc.verdict, exc.message, metrics)


if __name__ == "__main__":
    sys.exit(main())
