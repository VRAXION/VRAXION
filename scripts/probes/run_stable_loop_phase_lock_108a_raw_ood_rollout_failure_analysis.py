#!/usr/bin/env python3
"""Analysis-only raw OOD rollout failure attribution after 108."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_108A_RAW_OOD_ROLLOUT_FAILURE_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis/smoke")
DEFAULT_UPSTREAM_108_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_108_open_domain_assistant_ood_stress_and_failure_map_batch/smoke")

POSITIVE_VERDICT = "RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_POSITIVE"
BOUNDARY_TEXT = (
    "108A is analysis-only. It reads existing 108 OOD stress artifacts, performs no training, no repair, "
    "mutates no checkpoint, and changes no runtime/service/deploy surface. It is not GPT-like assistant "
    "readiness, not open-domain assistant readiness, not production chat, not public API, not deployment "
    "readiness, and not safety alignment."
)

REQUIRED_108_ARTIFACTS = [
    "summary.json",
    "multi_seed_aggregate.json",
    "checkpoint_integrity_manifest.json",
    "raw_generation_results.jsonl",
    "decoder_repaired_results.jsonl",
    "failure_mode_map.json",
    "raw_vs_decoder_ood_gap.json",
]
ALLOWED_SURFACE_LABELS = {
    "first_token_wrong",
    "case_id_drift",
    "slot_value_drift",
    "distractor_leak",
    "stale_or_old_value_used",
    "instruction_boundary_lost",
    "unsupported_answered_instead_of_refused",
    "over_refusal",
    "under_refusal",
    "hallucinated_fact",
    "prompt_injection_followed",
    "long_context_derailment",
    "adversarial_format_derailment",
    "stop_condition_failure",
    "repetition_or_loop",
    "malformed_utf8_or_garbled_output",
    "wrong_language",
    "unknown_raw_failure",
}
ALLOWED_MECHANISM_LABELS = {
    "prefix_loss_or_rollout_drift",
    "decoder_policy_gap",
    "context_carry_failure",
    "instruction_boundary_loss",
    "prompt_format_sensitivity",
    "stop_condition_weakness",
    "sft_coverage_gap",
    "wrong_language_drift",
    "unknown_mechanism",
}
FAMILY_SURFACE = {
    "OOD_PROVIDED_FACT_DISTRACTOR_TRAP": "distractor_leak",
    "OOD_AMBIGUOUS_INSTRUCTION": "instruction_boundary_lost",
    "OOD_CONFLICTING_INSTRUCTION": "instruction_boundary_lost",
    "OOD_LONG_NOISY_CONTEXT": "long_context_derailment",
    "OOD_MULTI_TURN_CORRECTION": "slot_value_drift",
    "OOD_MULTI_TURN_STALE_OVERRIDE": "stale_or_old_value_used",
    "OOD_PROMPT_INJECTION_ROLEPLAY": "prompt_injection_followed",
    "OOD_PROMPT_INJECTION_FORMAT_TRAP": "prompt_injection_followed",
    "OOD_ADVERSARIAL_FORMATTING": "adversarial_format_derailment",
    "OOD_WRONG_LANGUAGE_TRAP": "wrong_language",
    "OOD_HUNGARIAN_DIAGNOSTIC": "wrong_language",
    "OOD_UNSUPPORTED_WORLD_KNOWLEDGE": "unsupported_answered_instead_of_refused",
    "OOD_HALLUCINATION_INSUFFICIENT_FACTS": "hallucinated_fact",
    "OOD_OVER_REFUSAL_CHECK": "over_refusal",
    "OOD_UNDER_REFUSAL_CHECK": "under_refusal",
}
FAMILY_MECHANISM = {
    "OOD_PROVIDED_FACT_DISTRACTOR_TRAP": "decoder_policy_gap",
    "OOD_AMBIGUOUS_INSTRUCTION": "instruction_boundary_loss",
    "OOD_CONFLICTING_INSTRUCTION": "instruction_boundary_loss",
    "OOD_LONG_NOISY_CONTEXT": "context_carry_failure",
    "OOD_MULTI_TURN_CORRECTION": "context_carry_failure",
    "OOD_MULTI_TURN_STALE_OVERRIDE": "context_carry_failure",
    "OOD_PROMPT_INJECTION_ROLEPLAY": "instruction_boundary_loss",
    "OOD_PROMPT_INJECTION_FORMAT_TRAP": "prompt_format_sensitivity",
    "OOD_ADVERSARIAL_FORMATTING": "prompt_format_sensitivity",
    "OOD_WRONG_LANGUAGE_TRAP": "wrong_language_drift",
    "OOD_HUNGARIAN_DIAGNOSTIC": "wrong_language_drift",
    "OOD_UNSUPPORTED_WORLD_KNOWLEDGE": "decoder_policy_gap",
    "OOD_HALLUCINATION_INSUFFICIENT_FACTS": "decoder_policy_gap",
    "OOD_OVER_REFUSAL_CHECK": "decoder_policy_gap",
    "OOD_UNDER_REFUSAL_CHECK": "decoder_policy_gap",
}


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
        raise GateError("RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_FAILS", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_FAILS", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def resolve_upstream(text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


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
    if count == len(expected_tokens) and len(observed_tokens) == len(expected_tokens):
        first_wrong: int | None = None
    else:
        first_wrong = count
    expected_count = len(expected_tokens)
    return {
        "first_wrong_token_position": first_wrong,
        "matching_prefix_token_count": count,
        "expected_token_count": expected_count,
        "gold_prefix_survival_rate": count / expected_count if expected_count else 1.0,
        "divergence_point": {
            "expected_token": expected_tokens[count] if count < len(expected_tokens) else None,
            "raw_token": observed_tokens[count] if count < len(observed_tokens) else None,
        },
    }


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d{4,}\b", text))


def extract_after_markers(text: str) -> set[str]:
    return set(re.findall(r"\b(?:marker|code|tool|topic|slot|value)\s+([a-z][a-z0-9_-]*)", text.lower()))


def surface_label(raw: dict[str, Any], decoder: dict[str, Any], prefix: dict[str, Any]) -> str:
    family = raw.get("eval_family", "")
    expected = raw.get("expected_response", "")
    raw_text = raw.get("generated_text", "")
    prompt = raw.get("prompt", "")
    if raw.get("utf8_valid") is False:
        return "malformed_utf8_or_garbled_output"
    if raw.get("repetition_flag") is True:
        return "repetition_or_loop"
    if not raw.get("nonempty", True):
        return "stop_condition_failure"
    if raw.get("invented_fact_flag") is True:
        return "hallucinated_fact"
    if family in FAMILY_SURFACE:
        return FAMILY_SURFACE[family]
    expected_numbers = extract_numbers(expected) or extract_numbers(prompt)
    raw_numbers = extract_numbers(raw_text)
    if expected_numbers and raw_numbers and not (expected_numbers & raw_numbers):
        return "case_id_drift"
    expected_slots = extract_after_markers(expected)
    raw_slots = extract_after_markers(raw_text)
    if expected_slots and raw_slots and not (expected_slots & raw_slots):
        return "slot_value_drift"
    forbidden = [item.lower() for item in raw.get("forbidden_outputs", [])]
    if any(item and item in raw_text.lower() for item in forbidden):
        return "distractor_leak"
    if prefix.get("first_wrong_token_position") == 0:
        return "first_token_wrong"
    return "unknown_raw_failure"


def mechanism_label(raw: dict[str, Any], surface: str, prefix: dict[str, Any]) -> str:
    family = raw.get("eval_family", "")
    if family in FAMILY_MECHANISM:
        return FAMILY_MECHANISM[family]
    if surface in {"case_id_drift", "slot_value_drift", "distractor_leak", "stale_or_old_value_used", "first_token_wrong"}:
        return "prefix_loss_or_rollout_drift"
    if surface in {"instruction_boundary_lost", "unsupported_answered_instead_of_refused", "under_refusal", "over_refusal", "prompt_injection_followed"}:
        return "instruction_boundary_loss"
    if surface == "stop_condition_failure":
        return "stop_condition_weakness"
    if surface == "adversarial_format_derailment":
        return "prompt_format_sensitivity"
    if surface == "wrong_language":
        return "wrong_language_drift"
    if prefix.get("gold_prefix_survival_rate", 1.0) < 0.40:
        return "prefix_loss_or_rollout_drift"
    return "unknown_mechanism"


def write_summary(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    payload = {
        "schema_version": "raw_ood_rollout_failure_analysis_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "phase": status,
        "analysis_only": True,
        "model_capability_improved_by_108a": False,
        "training_performed": False,
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
        "108A is analysis-only. It does not train, repair, mutate checkpoints, or change runtime/service/deploy code.",
        "It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.",
        "",
        f"Status: `{summary.get('status')}`",
        "",
        "## Key Metrics",
        "",
        f"- raw_failure_count: `{metrics.get('raw_failure_count')}`",
        f"- raw_decoder_disagreement_count: `{metrics.get('raw_decoder_disagreement_count')}`",
        f"- decoder_success_on_raw_fail_rate: `{metrics.get('decoder_success_on_raw_fail_rate')}`",
        f"- unknown_raw_failure_rate: `{metrics.get('unknown_raw_failure_rate')}`",
        f"- raw_rollout_drift_rate: `{metrics.get('raw_rollout_drift_rate')}`",
        f"- first_wrong_token_position_mean: `{metrics.get('first_wrong_token_position_mean')}`",
        f"- gold_prefix_survival_rate_mean: `{metrics.get('gold_prefix_survival_rate_mean')}`",
        "",
        "## Recommendation",
        "",
        f"- next: `{metrics.get('recommended_next')}`",
        f"- primary_failure_mechanism: `{metrics.get('primary_failure_mechanism')}`",
        "",
        "## Verdicts",
        "",
    ]
    lines.extend(f"- `{verdict}`" for verdict in summary.get("verdicts", []))
    write_text(out / "report.md", "\n".join(lines) + "\n")


def fail(out: Path, verdict: str, message: str, metrics: dict[str, Any]) -> int:
    append_progress(out, "failure", "failed", verdict=verdict, message=message)
    write_summary(out, "failed", ["RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_FAILS", verdict], metrics, message)
    return 1


def require_upstream_108(root: Path) -> dict[str, Any]:
    for rel_name in REQUIRED_108_ARTIFACTS:
        if not (root / rel_name).exists():
            raise GateError("UPSTREAM_108_ARTIFACT_MISSING", f"missing 108 artifact: {rel_name}")
    summary = read_json(root / "summary.json")
    verdicts = set(summary.get("verdicts", []))
    if "OPEN_DOMAIN_ASSISTANT_OOD_STRESS_AND_FAILURE_MAP_BATCH_POSITIVE" not in verdicts:
        raise GateError("UPSTREAM_108_NOT_POSITIVE", "108 positive verdict missing")
    metrics = summary.get("metrics", {})
    required = {
        "checkpoint_hash_unchanged": True,
        "bounded_release_artifact_unchanged": True,
        "artifact_exfiltration_count": 0,
        "gpt_like_claim_count": 0,
        "production_chat_claim_count": 0,
        "train_step_count": 0,
        "optimizer_step_count": 0,
    }
    for key, expected in required.items():
        if metrics.get(key) != expected:
            raise GateError("UPSTREAM_108_NOT_POSITIVE", f"108 metric {key} expected {expected!r}, got {metrics.get(key)!r}")
    if metrics.get("raw_ood_stress_accuracy", 1.0) > 0.60:
        raise GateError("UPSTREAM_108_NOT_POSITIVE", "108 raw OOD stress gap is not present")
    if metrics.get("decoder_ood_stress_accuracy") != 1.0:
        raise GateError("UPSTREAM_108_NOT_POSITIVE", "108 decoder OOD accuracy is not 1.0")
    if metrics.get("raw_vs_decoder_ood_gap", 0.0) < 0.25:
        raise GateError("UPSTREAM_108_NOT_POSITIVE", "108 raw/decoder gap too small")
    if metrics.get("unknown_failure_rate", 1.0) > 0.10:
        raise GateError("UPSTREAM_108_NOT_POSITIVE", "108 unknown failure rate too high")
    return summary


def pair_rows(raw_rows: list[dict[str, Any]], decoder_rows: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[str]]:
    failures: list[str] = []
    decoder_by_key = {(row.get("seed"), row.get("eval_index"), row.get("eval_family")): row for row in decoder_rows}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw in raw_rows:
        key = (raw.get("seed"), raw.get("eval_index"), raw.get("eval_family"))
        decoder = decoder_by_key.get(key)
        if decoder is None:
            failures.append(f"missing_decoder_pair:{key}")
            continue
        pairs.append((raw, decoder))
    if len(pairs) != len(raw_rows) or len(decoder_by_key) != len(decoder_rows):
        failures.append("pair_count_mismatch")
    return pairs, failures


def make_attribution(raw: dict[str, Any], decoder: dict[str, Any]) -> dict[str, Any]:
    expected = str(raw.get("expected_response") or decoder.get("expected_response") or "")
    raw_text = str(raw.get("generated_text") or "")
    decoder_text = str(decoder.get("generated_text") or "")
    prefix = first_wrong_position(expected, raw_text)
    surface = surface_label(raw, decoder, prefix)
    mechanism = mechanism_label(raw, surface, prefix)
    if surface not in ALLOWED_SURFACE_LABELS:
        surface = "unknown_raw_failure"
    if mechanism not in ALLOWED_MECHANISM_LABELS:
        mechanism = "unknown_mechanism"
    return {
        "schema_version": "raw_ood_failure_attribution_row_v1",
        "seed": raw.get("seed"),
        "eval_index": raw.get("eval_index"),
        "eval_family": raw.get("eval_family"),
        "prompt": raw.get("prompt"),
        "expected_response_source": "upstream_108_expected_response",
        "expected_response": expected,
        "decoder_output": decoder_text,
        "raw_output": raw_text,
        "decoder_pass_fail": decoder.get("pass_fail"),
        "raw_pass_fail": raw.get("pass_fail"),
        "upstream_108_failure_label": raw.get("failure_label"),
        "primary_surface_failure_label": surface,
        "likely_mechanism_label": mechanism,
        "divergence_point": prefix["divergence_point"],
        "first_wrong_token_position": prefix["first_wrong_token_position"],
        "matching_prefix_token_count": prefix["matching_prefix_token_count"],
        "expected_token_count": prefix["expected_token_count"],
        "gold_prefix_survival_rate": prefix["gold_prefix_survival_rate"],
        "required_keywords": raw.get("required_keywords", []),
        "forbidden_outputs": raw.get("forbidden_outputs", []),
        "short_diagnosis": f"raw failed while decoder passed; surface={surface}; mechanism={mechanism}",
        "llm_judge_used": False,
        "prediction_oracle_used": False,
    }


def aggregate_metrics(pairs: list[tuple[dict[str, Any], dict[str, Any]]], attributions: list[dict[str, Any]], checkpoint: dict[str, Any], gap: dict[str, Any]) -> dict[str, Any]:
    raw_failures = [raw for raw, _decoder in pairs if raw.get("pass_fail") == "fail"]
    disagreements = [item for item in attributions if item.get("raw_pass_fail") == "fail" and item.get("decoder_pass_fail") == "pass"]
    first_positions = [item["first_wrong_token_position"] for item in disagreements if item.get("first_wrong_token_position") is not None]
    prefix_rates = [float(item.get("gold_prefix_survival_rate", 0.0)) for item in disagreements]
    surface_counts = Counter(item["primary_surface_failure_label"] for item in disagreements)
    mechanism_counts = Counter(item["likely_mechanism_label"] for item in disagreements)
    unknown_count = surface_counts.get("unknown_raw_failure", 0) + mechanism_counts.get("unknown_mechanism", 0)
    raw_failure_count = len(raw_failures)
    disagreement_count = len(disagreements)
    return {
        "schema_version": "raw_ood_rollout_failure_analysis_metrics_v1",
        "total_pair_count": len(pairs),
        "raw_failure_count": raw_failure_count,
        "raw_decoder_disagreement_count": disagreement_count,
        "decoder_success_on_raw_fail_rate": disagreement_count / raw_failure_count if raw_failure_count else 0.0,
        "raw_rollout_drift_rate": disagreement_count / len(pairs) if pairs else 0.0,
        "unknown_raw_failure_rate": unknown_count / max(1, disagreement_count),
        "first_wrong_token_position_mean": statistics.fmean(first_positions) if first_positions else None,
        "first_wrong_token_position_median": statistics.median(first_positions) if first_positions else None,
        "gold_prefix_survival_rate_mean": statistics.fmean(prefix_rates) if prefix_rates else None,
        "gold_prefix_survival_rate_min": min(prefix_rates) if prefix_rates else None,
        "case_id_drift_rate": surface_counts.get("case_id_drift", 0) / max(1, disagreement_count),
        "slot_drift_rate": surface_counts.get("slot_value_drift", 0) / max(1, disagreement_count),
        "distractor_leak_rate": surface_counts.get("distractor_leak", 0) / max(1, disagreement_count),
        "stale_value_rate": surface_counts.get("stale_or_old_value_used", 0) / max(1, disagreement_count),
        "hallucinated_fact_rate": surface_counts.get("hallucinated_fact", 0) / max(1, disagreement_count),
        "over_refusal_rate": surface_counts.get("over_refusal", 0) / max(1, disagreement_count),
        "under_refusal_rate": surface_counts.get("under_refusal", 0) / max(1, disagreement_count),
        "prompt_injection_follow_rate": surface_counts.get("prompt_injection_followed", 0) / max(1, disagreement_count),
        "stop_condition_failure_rate": surface_counts.get("stop_condition_failure", 0) / max(1, disagreement_count),
        "repetition_rate": surface_counts.get("repetition_or_loop", 0) / max(1, disagreement_count),
        "utf8_valid_rate": sum(1 for raw, _ in pairs if raw.get("utf8_valid") is not False) / max(1, len(pairs)),
        "raw_ood_stress_accuracy": gap.get("raw_ood_stress_accuracy"),
        "decoder_ood_stress_accuracy": gap.get("decoder_ood_stress_accuracy"),
        "raw_vs_decoder_ood_gap": gap.get("raw_vs_decoder_ood_gap"),
        "checkpoint_hash_unchanged": checkpoint.get("checkpoint_hash_unchanged") is True,
        "bounded_release_artifact_unchanged": checkpoint.get("bounded_release_artifact_unchanged") is True,
        "train_step_count": checkpoint.get("train_step_count", 0),
        "optimizer_step_count": checkpoint.get("optimizer_step_count", 0),
        "surface_failure_counts": dict(surface_counts),
        "mechanism_failure_counts": dict(mechanism_counts),
    }


def make_repair_plan(metrics: dict[str, Any]) -> dict[str, Any]:
    mechanism_counts = Counter(metrics.get("mechanism_failure_counts", {}))
    surface_counts = Counter(metrics.get("surface_failure_counts", {}))
    total = max(1, int(metrics.get("raw_decoder_disagreement_count", 0)))
    primary_mechanism = mechanism_counts.most_common(1)[0][0] if mechanism_counts else "unknown_mechanism"
    secondary = [name for name, _count in mechanism_counts.most_common()[1:4]]
    decoder_success = float(metrics.get("decoder_success_on_raw_fail_rate", 0.0))
    gap = float(metrics.get("raw_vs_decoder_ood_gap", 0.0))
    if decoder_success >= 0.95 and gap >= 0.25:
        next_step = "109_DECODER_POLICY_INTEGRATION"
        secondary_next = "109_RAW_ROLLOUT_REPAIR" if surface_counts.get("first_token_wrong", 0) / total >= 0.25 else "109_SFT_ROLLOUT_DATA_REPAIR"
    elif (surface_counts.get("first_token_wrong", 0) + mechanism_counts.get("prefix_loss_or_rollout_drift", 0)) / total >= 0.40:
        next_step = "109_RAW_ROLLOUT_REPAIR"
        secondary_next = "109_SFT_ROLLOUT_DATA_REPAIR"
    elif mechanism_counts.get("stop_condition_weakness", 0) / total >= 0.30:
        next_step = "109_STOP_CONDITION_REPAIR"
        secondary_next = "109_RAW_ROLLOUT_REPAIR"
    elif mechanism_counts.get("prompt_format_sensitivity", 0) / total >= 0.30:
        next_step = "109_PROMPT_FORMAT_REPAIR"
        secondary_next = "109_SFT_ROLLOUT_DATA_REPAIR"
    else:
        next_step = "109_SFT_ROLLOUT_DATA_REPAIR"
        secondary_next = "109_RAW_ROLLOUT_REPAIR"
    return {
        "schema_version": "raw_ood_rollout_recommended_repair_plan_v1",
        "next": next_step,
        "secondary_next_if_decoder_integration_fails": secondary_next,
        "primary_failure_mechanism": primary_mechanism,
        "secondary_failure_mechanisms": secondary,
        "evidence_counts": {
            "surface_failure_counts": metrics.get("surface_failure_counts", {}),
            "mechanism_failure_counts": metrics.get("mechanism_failure_counts", {}),
            "raw_decoder_disagreement_count": metrics.get("raw_decoder_disagreement_count"),
            "raw_failure_count": metrics.get("raw_failure_count"),
        },
        "evidence_rates": {
            "decoder_success_on_raw_fail_rate": metrics.get("decoder_success_on_raw_fail_rate"),
            "raw_vs_decoder_ood_gap": metrics.get("raw_vs_decoder_ood_gap"),
            "unknown_raw_failure_rate": metrics.get("unknown_raw_failure_rate"),
            "raw_rollout_drift_rate": metrics.get("raw_rollout_drift_rate"),
        },
        "recommended_scope_for_109": "Integrate the deterministic decoder policy that already passes 108 OOD stress, then re-check raw/decoder parity without mutating the bounded release stack.",
    }


def write_reports(out: Path, upstream_root: Path, summary_108: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_rows = read_jsonl(upstream_root / "raw_generation_results.jsonl")
    decoder_rows = read_jsonl(upstream_root / "decoder_repaired_results.jsonl")
    pairs, pair_failures = pair_rows(raw_rows, decoder_rows)
    if pair_failures:
        raise GateError("RAW_DECODER_PAIRING_FAILS", "; ".join(pair_failures[:5]))
    disagreements = [(raw, decoder) for raw, decoder in pairs if raw.get("pass_fail") == "fail" and decoder.get("pass_fail") == "pass"]
    if not pairs or not disagreements:
        raise GateError("RAW_DECODER_PAIRING_FAILS", "expected raw-fail / decoder-pass disagreements")
    attributions = [make_attribution(raw, decoder) for raw, decoder in disagreements]
    checkpoint = read_json(upstream_root / "checkpoint_integrity_manifest.json")
    gap = read_json(upstream_root / "raw_vs_decoder_ood_gap.json")
    metrics = aggregate_metrics(pairs, attributions, checkpoint, gap)
    repair_plan = make_repair_plan(metrics)
    metrics["recommended_next"] = repair_plan["next"]
    metrics["primary_failure_mechanism"] = repair_plan["primary_failure_mechanism"]

    surface_counts = Counter(item["primary_surface_failure_label"] for item in attributions)
    mechanism_counts = Counter(item["likely_mechanism_label"] for item in attributions)
    family_counts = Counter(item["eval_family"] for item in attributions)
    first_positions = [item["first_wrong_token_position"] for item in attributions if item["first_wrong_token_position"] is not None]
    prefix_rates = [item["gold_prefix_survival_rate"] for item in attributions]

    write_json(out / "raw_decoder_pair_manifest.json", {
        "schema_version": "raw_ood_rollout_pair_manifest_v1",
        "raw_pair_count": len(raw_rows),
        "decoder_pair_count": len(decoder_rows),
        "paired_row_count": len(pairs),
        "raw_decoder_disagreement_count": len(disagreements),
        "pair_key": ["seed", "eval_index", "eval_family"],
    })
    write_json(out / "raw_failure_attribution.json", {
        "schema_version": "raw_ood_rollout_failure_attribution_v1",
        "all_raw_fail_decoder_pass_rows_attributed": len(attributions) == len(disagreements),
        "unknown_raw_failure_rate": metrics["unknown_raw_failure_rate"],
        "surface_failure_counts": dict(surface_counts),
        "mechanism_failure_counts": dict(mechanism_counts),
        "allowed_surface_labels": sorted(ALLOWED_SURFACE_LABELS),
        "allowed_mechanism_labels": sorted(ALLOWED_MECHANISM_LABELS),
    })
    write_jsonl(out / "raw_failure_cases.jsonl", attributions)
    write_jsonl(out / "raw_decoder_disagreement.jsonl", attributions)
    write_json(out / "first_error_position_report.json", {
        "schema_version": "raw_ood_first_error_position_v1",
        "first_wrong_token_position_mean": metrics["first_wrong_token_position_mean"],
        "first_wrong_token_position_median": metrics["first_wrong_token_position_median"],
        "first_wrong_token_positions": first_positions,
    })
    write_json(out / "prefix_survival_report.json", {
        "schema_version": "raw_ood_prefix_survival_v1",
        "gold_prefix_survival_rate_mean": metrics["gold_prefix_survival_rate_mean"],
        "gold_prefix_survival_rate_min": metrics["gold_prefix_survival_rate_min"],
        "gold_prefix_survival_rates": prefix_rates,
    })
    write_json(out / "rollout_drift_report.json", {
        "schema_version": "raw_ood_rollout_drift_v1",
        "raw_rollout_drift_rate": metrics["raw_rollout_drift_rate"],
        "raw_decoder_disagreement_count": metrics["raw_decoder_disagreement_count"],
        "total_pair_count": metrics["total_pair_count"],
        "decoder_success_on_raw_fail_rate": metrics["decoder_success_on_raw_fail_rate"],
    })
    write_json(out / "stop_condition_report.json", {
        "schema_version": "raw_ood_stop_condition_v1",
        "stop_condition_failure_rate": metrics["stop_condition_failure_rate"],
        "stop_condition_failure_count": surface_counts.get("stop_condition_failure", 0),
        "repetition_rate": metrics["repetition_rate"],
    })
    write_json(out / "family_failure_breakdown.json", {
        "schema_version": "raw_ood_family_failure_breakdown_v1",
        "family_failure_counts": dict(family_counts),
        "surface_by_family": {
            family: dict(Counter(item["primary_surface_failure_label"] for item in attributions if item["eval_family"] == family))
            for family in sorted(family_counts)
        },
        "mechanism_by_family": {
            family: dict(Counter(item["likely_mechanism_label"] for item in attributions if item["eval_family"] == family))
            for family in sorted(family_counts)
        },
    })
    write_json(out / "recommended_repair_plan.json", repair_plan)
    samples = make_samples(attributions)
    write_jsonl(out / "human_readable_samples.jsonl", samples)
    write_json(out / "upstream_108_manifest.json", {
        "schema_version": "raw_ood_upstream_108_manifest_v1",
        "upstream_root": rel(upstream_root),
        "upstream_status": summary_108.get("status"),
        "upstream_verdicts": summary_108.get("verdicts", []),
        "upstream_metrics": summary_108.get("metrics", {}),
    })
    return metrics, repair_plan


def make_samples(attributions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    preferred_families = [
        "OOD_LONG_NOISY_CONTEXT",
        "OOD_PROMPT_INJECTION_ROLEPLAY",
        "OOD_PROMPT_INJECTION_FORMAT_TRAP",
        "OOD_UNSUPPORTED_WORLD_KNOWLEDGE",
        "OOD_PROVIDED_FACT_DISTRACTOR_TRAP",
        "OOD_MULTI_TURN_STALE_OVERRIDE",
        "OOD_ADVERSARIAL_FORMATTING",
        "OOD_HUNGARIAN_DIAGNOSTIC",
    ]
    for family in preferred_families:
        for item in attributions:
            if item["eval_family"] == family and family not in seen:
                samples.append(sample_row(item))
                seen.add(family)
                break
    for item in attributions:
        label = item["primary_surface_failure_label"]
        if label not in seen:
            samples.append(sample_row(item))
            seen.add(label)
        if len(samples) >= 24:
            break
    return samples


def sample_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": item["seed"],
        "eval_family": item["eval_family"],
        "prompt": item["prompt"],
        "raw_output": item["raw_output"],
        "decoder_output": item["decoder_output"],
        "expected_response": item["expected_response"],
        "primary_surface_failure_label": item["primary_surface_failure_label"],
        "likely_mechanism_label": item["likely_mechanism_label"],
        "first_wrong_token_position": item["first_wrong_token_position"],
        "gold_prefix_survival_rate": item["gold_prefix_survival_rate"],
        "short_diagnosis": item["short_diagnosis"],
    }


def validate_positive(metrics: dict[str, Any], repair_plan: dict[str, Any], out: Path) -> None:
    if metrics.get("raw_decoder_disagreement_count", 0) <= 0:
        raise GateError("RAW_FAILURE_ATTRIBUTION_INCOMPLETE", "no raw/decoder disagreements")
    if metrics.get("unknown_raw_failure_rate", 1.0) > 0.10:
        raise GateError("UNKNOWN_RAW_FAILURE_RATE_TOO_HIGH", "unknown raw failure rate too high")
    if not (out / "human_readable_samples.jsonl").exists():
        raise GateError("HUMAN_SAMPLE_REPORT_MISSING", "human samples missing")
    if not repair_plan.get("next"):
        raise GateError("REPAIR_PLAN_MISSING", "repair plan missing next")
    if metrics.get("checkpoint_hash_unchanged") is not True:
        raise GateError("CHECKPOINT_MUTATION_DETECTED", "checkpoint hash changed")
    if metrics.get("bounded_release_artifact_unchanged") is not True:
        raise GateError("CHECKPOINT_MUTATION_DETECTED", "bounded release changed")
    if metrics.get("train_step_count") != 0 or metrics.get("optimizer_step_count") != 0:
        raise GateError("TRAINING_SIDE_EFFECT_DETECTED", "training side effect detected")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run STABLE_LOOP_PHASE_LOCK_108A raw OOD rollout failure analysis")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-108-root", default=str(DEFAULT_UPSTREAM_108_ROOT))
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    args = parser.parse_args(argv)

    out = resolve_target_out(args.out)
    upstream_root = resolve_upstream(args.upstream_108_root)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "progress.jsonl").exists():
        (out / "progress.jsonl").unlink()

    start = time.time()
    metrics: dict[str, Any] = {
        "schema_version": "raw_ood_rollout_failure_analysis_metrics_v1",
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "checkpoint_hash_unchanged": None,
        "bounded_release_artifact_unchanged": None,
    }
    write_json(out / "queue.json", {
        "schema_version": "raw_ood_rollout_failure_analysis_queue_v1",
        "milestone": MILESTONE,
        "partial_write_policy": "progress summary report are written from start and refreshed after each phase",
        "steps": ["verify_upstream_108", "pair_rows", "attribute_failures", "write_reports", "repair_plan", "final"],
    })
    write_json(out / "analysis_config.json", {
        "schema_version": "raw_ood_rollout_failure_analysis_config_v1",
        "milestone": MILESTONE,
        "analysis_only": True,
        "upstream_108_root": rel(upstream_root),
        "tokenization": "lowercase alnum/underscore chunks plus punctuation tokens",
        "heartbeat_sec": args.heartbeat_sec,
        "llm_judge_used": False,
        "prediction_oracle_used": False,
        "boundary": BOUNDARY_TEXT,
    })
    append_progress(out, "start", "running", milestone=MILESTONE)
    write_summary(out, "running", ["RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_RUNNING"], metrics)

    try:
        summary_108 = require_upstream_108(upstream_root)
        metrics.update({
            "upstream_108_status": summary_108.get("status"),
            "upstream_raw_ood_stress_accuracy": summary_108.get("metrics", {}).get("raw_ood_stress_accuracy"),
            "upstream_decoder_ood_stress_accuracy": summary_108.get("metrics", {}).get("decoder_ood_stress_accuracy"),
            "upstream_raw_vs_decoder_ood_gap": summary_108.get("metrics", {}).get("raw_vs_decoder_ood_gap"),
        })
        append_progress(out, "upstream verification", "completed", upstream_108_root=rel(upstream_root))
        write_summary(out, "running", ["UPSTREAM_108_STRESS_MAP_VERIFIED"], metrics)

        metrics, repair_plan = write_reports(out, upstream_root, summary_108)
        metrics["wall_clock_sec"] = round(time.time() - start, 3)
        append_progress(out, "failure attribution", "completed", disagreements=metrics["raw_decoder_disagreement_count"])
        write_summary(out, "running", ["RAW_FAILURES_ATTRIBUTED"], metrics)

        validate_positive(metrics, repair_plan, out)
        append_progress(out, "repair plan", "completed", next=repair_plan["next"])
        write_summary(out, "running", ["REPAIR_PLAN_WRITTEN"], metrics)

        verdicts = [
            POSITIVE_VERDICT,
            "UPSTREAM_108_STRESS_MAP_VERIFIED",
            "RAW_DECODER_GAP_CONFIRMED",
            "RAW_FAILURES_ATTRIBUTED",
            "PREFIX_SURVIVAL_ANALYZED",
            "ROLLOUT_DRIFT_ANALYZED",
            "STOP_CONDITION_ANALYZED",
            "REPAIR_PLAN_WRITTEN",
            "NO_TRAINING_PERFORMED",
            "CHECKPOINT_UNCHANGED",
            "GPT_LIKE_READINESS_NOT_CLAIMED",
            "PRODUCTION_CHAT_NOT_CLAIMED",
        ]
        append_progress(out, "final verdict", "positive", next=repair_plan["next"])
        write_summary(out, "positive", verdicts, metrics)
        print(POSITIVE_VERDICT)
        print(json.dumps({"out": rel(out), "next": repair_plan["next"], "raw_decoder_disagreement_count": metrics["raw_decoder_disagreement_count"]}, sort_keys=True))
        return 0
    except GateError as exc:
        return fail(out, exc.verdict, exc.message, metrics)


if __name__ == "__main__":
    sys.exit(main())
