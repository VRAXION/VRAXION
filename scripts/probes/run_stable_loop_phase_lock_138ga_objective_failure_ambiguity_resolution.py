#!/usr/bin/env python3
"""138GA artifact-only near-match ambiguity resolution.

This phase reads existing 138G and 138R artifacts only. It does not train, run
new model inference, call shared_raw_generation_helper.py, run torch forward
passes, mutate checkpoints, modify helper/backend code, import old runners,
start services, deploy, delete or consolidate files, or modify runtime,
release, or product surfaces.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138ga_objective_failure_ambiguity_resolution/smoke")
DEFAULT_UPSTREAM_138G_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138g_real_raw_reasoning_objective_failure_analysis/smoke")
DEFAULT_UPSTREAM_138R_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138r_real_raw_reasoning_repair_training_plan_or_probe/smoke")
BOUNDARY_TEXT = (
    "138GA is artifact-only near-match ambiguity resolution. It does not "
    "train, run new model inference, call shared_raw_generation_helper.py, "
    "run torch forward passes, mutate checkpoints, modify helper/backend "
    "code, import old runners, start services, deploy, delete or consolidate "
    "files, modify runtime/release/product surfaces, or change root LICENSE. "
    "It does not restore reasoning, raw assistant capability, structured/tool "
    "capability, GPT-like readiness, open-domain readiness, production chat, "
    "public API, deployment readiness, or safety alignment."
)
FALSE_FLAGS = {
    "reasoning_restored": False,
    "reasoning_subtrack_real_raw_evidence_partially_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "gpt_like_readiness_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
}
PRIMARY_LABELS = [
    "meaningful_partial_answer",
    "numeric_partial_match",
    "formatting_or_wrapper_mismatch",
    "train_namespace_overlap",
    "stale_chat_overlap",
    "prompt_copy_overlap",
    "distractor_overlap",
    "common_token_overlap",
    "scorer_false_near_match",
    "unknown_near_match",
]
TRIVIAL_LABELS = {
    "train_namespace_overlap",
    "stale_chat_overlap",
    "prompt_copy_overlap",
    "distractor_overlap",
    "common_token_overlap",
    "scorer_false_near_match",
}
NONTRIVIAL_LABELS = {
    "meaningful_partial_answer",
    "numeric_partial_match",
    "formatting_or_wrapper_mismatch",
}
COMMON_TOKENS = {
    "answer",
    "case",
    "value",
    "marker",
    "token",
    "return",
    "row",
    "repair",
    "prefix",
    "split",
    "fact",
}
REQUIRED_138G_ARTIFACTS = [
    "decision.json",
    "summary.json",
    "scoring_strictness_recheck.json",
    "rollout_output_pattern_report.json",
    "first_mismatch_report.json",
    "prompt_answer_alignment_report.json",
    "teacher_forcing_vs_rollout_report.json",
    "objective_failure_root_cause.json",
    "diagnostic_gap_register.json",
]
REQUIRED_138R_ARTIFACTS = [
    "scoring_results.jsonl",
    "raw_generation_results.jsonl",
    "eval_rows.jsonl",
    "raw_generation_trace.jsonl",
    "aggregate_metrics.json",
    "control_arm_report.json",
    "freshness_leakage_audit.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "helper_provenance_verification.json",
    "generated_before_scoring_report.json",
]


class GateError(Exception):
    def __init__(self, verdict: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.verdict = verdict
        self.message = message
        self.details = details or {}


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise GateError("138GA_BOUNDARY_FAILURE", "--out must stay inside the repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138GA_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    if any(part == ".." for part in relative.parts):
        raise GateError("138GA_BOUNDARY_FAILURE", "--out must not escape target/pilot_wave")
    return resolved


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def token_set(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text)}


def answer_body(expected: str) -> str:
    if "=" in expected:
        return expected.split("=", 1)[1]
    return expected


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def contains_numeric_token(text: str, digits: str) -> bool:
    return re.search(rf"(?<!\d){re.escape(digits)}(?!\d)", text) is not None


def train_namespace_overlap(generated: str, expected: str) -> bool:
    return expected.startswith("ANSWER=E") and re.search(r"\bANSWER=T[A-Za-z0-9_]*", generated) is not None


def prompt_copy_overlap(generated: str, prompt: str) -> bool:
    gen_tokens = token_set(generated)
    prompt_tokens = token_set(prompt)
    if not gen_tokens or not prompt_tokens:
        return False
    if generated.strip() and generated.strip() in prompt:
        return True
    overlap = len(gen_tokens & prompt_tokens) / max(1, len(gen_tokens))
    return overlap >= 0.65 and len(gen_tokens) >= 8


def stale_dominates(generated: str) -> bool:
    if not re.search(r"\b(User|Assistant):", generated):
        return False
    stale_lines = [line for line in generated.splitlines() if re.search(r"\b(User|Assistant):", line)]
    answer_lines = [line for line in generated.splitlines() if "ANSWER=" in line]
    return len(stale_lines) >= max(1, len(answer_lines))


def classify_row(score: dict[str, Any], raw: dict[str, Any], eval_row: dict[str, Any]) -> dict[str, Any]:
    row_id = score["row_id"]
    expected = str(score.get("expected_output") or eval_row.get("expected_output", ""))
    generated = str(raw.get("generated_text", ""))
    prompt = str(eval_row.get("prompt", ""))
    distractor = str(eval_row.get("forbidden_distractor", ""))
    body = answer_body(expected)
    digits = numeric_tokens(expected)
    secondary: list[str] = []
    reason = ""
    meaningful = False

    exact_expected_present = expected in generated
    correct_numeric_token_present = any(contains_numeric_token(generated, item) for item in digits)
    body_present = bool(body and body in generated)
    train_ns = train_namespace_overlap(generated, expected)
    stale = stale_dominates(generated)
    prompt_copy = prompt_copy_overlap(generated, prompt)
    distractor_present = bool(distractor and distractor in generated)
    common_overlap = bool(token_set(generated) & COMMON_TOKENS)
    digit_substring = bool(digits and any(item in generated for item in digits))

    if exact_expected_present:
        label = "meaningful_partial_answer"
        meaningful = True
        reason = "exact expected output appears in generated_text"
    elif train_ns:
        label = "train_namespace_overlap"
        reason = "generated_text emits ANSWER=T namespace while expected_output uses ANSWER=E namespace"
    elif correct_numeric_token_present and not train_ns:
        label = "numeric_partial_match"
        reason = "correct numeric answer appears as a standalone numeric token"
    elif body_present and not train_ns:
        label = "formatting_or_wrapper_mismatch"
        reason = "expected answer body appears without the required wrapper"
    elif stale:
        label = "stale_chat_overlap"
        reason = "stale User:/Assistant: fragment dominates generated_text"
    elif prompt_copy:
        label = "prompt_copy_overlap"
        reason = "generated_text mostly copies prompt tokens"
    elif distractor_present:
        label = "distractor_overlap"
        reason = "generated_text contains the row distractor token"
    elif common_overlap and not digit_substring:
        label = "common_token_overlap"
        reason = "near-match evidence is limited to common task tokens"
    elif digit_substring:
        label = "scorer_false_near_match"
        reason = "near_match is caused by weak expected-digit substring overlap"
    else:
        label = "unknown_near_match"
        reason = "near-match source is not deterministically classifiable"

    if correct_numeric_token_present:
        secondary.append("correct_numeric_token_present")
    if digit_substring and not correct_numeric_token_present:
        secondary.append("digit_substring_only")
    if train_ns and label != "train_namespace_overlap":
        secondary.append("train_namespace_overlap")
    if stale and label != "stale_chat_overlap":
        secondary.append("stale_chat_overlap")
    if prompt_copy and label != "prompt_copy_overlap":
        secondary.append("prompt_copy_overlap")
    if distractor_present and label != "distractor_overlap":
        secondary.append("distractor_overlap")
    if common_overlap and label != "common_token_overlap":
        secondary.append("common_token_overlap")
    if score.get("expected_token_included") is True:
        secondary.append("expected_token_included")

    if label == "meaningful_partial_answer":
        why = "meaningful: strict expected output criterion is satisfied"
    elif label == "numeric_partial_match":
        why = "not enough for restored reasoning: numeric token is present but exact expected output is absent"
    elif label == "formatting_or_wrapper_mismatch":
        why = "potential scorer/eval contribution: answer body appears without required wrapper"
    elif label == "train_namespace_overlap":
        why = "not meaningful by default: train namespace token appears where eval namespace token is required"
    elif label == "scorer_false_near_match":
        why = "not meaningful: weak near-match digit substring does not prove the correct answer"
    else:
        why = "not meaningful under strict deterministic 138GA criteria"

    return {
        "row_id": row_id,
        "family": score.get("family", eval_row.get("family")),
        "seed": score.get("seed", eval_row.get("seed")),
        "prompt": prompt,
        "generated_text": generated,
        "expected_output": expected,
        "expected_output_hash": score.get("expected_output_hash"),
        "generated_text_hash": raw.get("generated_text_hash", score.get("generated_text_hash")),
        "helper_trace_hash": score.get("helper_trace_hash", raw.get("generation_trace_hash")),
        "near_match": True,
        "near_match_score/source": "138R scoring_results.jsonl.near_match",
        "near_match_score_source": "138R scoring_results.jsonl.near_match",
        "primary_label": label,
        "primary_classification": label,
        "secondary_labels": sorted(set(secondary)),
        "deterministic_reason": reason,
        "why_meaningful_or_not": why,
        "strict_meaningful_partial_answer": meaningful,
        "exact_expected_present": exact_expected_present,
        "correct_numeric_token_present": correct_numeric_token_present,
        "expected_answer_body_present": body_present,
        "train_namespace_overlap": train_ns,
        "stale_chat_overlap": stale,
        "prompt_copy_overlap": prompt_copy,
        "distractor_overlap": distractor_present,
        "common_token_overlap": common_overlap,
        "digit_substring_overlap": digit_substring,
    }


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138ga_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "artifact_only_analysis": True,
            "training_performed": False,
            "new_model_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            "checkpoint_mutated": False,
            "runtime_surface_mutated": False,
            "root_license_changed": False,
            **FALSE_FLAGS,
            "metrics": decision,
        },
    )


def write_report(out: Path, verdicts: list[str], decision: dict[str, Any]) -> None:
    lines = [
        f"# {MILESTONE}",
        "",
        "## Boundary",
        "",
        BOUNDARY_TEXT,
        "",
        "## Verdicts",
        "",
    ]
    lines.extend(f"- `{verdict}`" for verdict in verdicts)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `decision`: `{decision.get('decision')}`",
            f"- `next`: `{decision.get('next')}`",
            f"- `near_match_row_count`: `{decision.get('near_match_row_count')}`",
            f"- `near_match_rate`: `{decision.get('near_match_rate')}`",
            f"- `meaningful_near_match_rate`: `{decision.get('meaningful_near_match_rate')}`",
            f"- `unknown_near_match_rate`: `{decision.get('unknown_near_match_rate')}`",
            "",
            "138GA is artifact-only near-match disambiguation.",
            "Reasoning is not restored.",
            "The reasoning subtrack real-raw evidence is not partially restored.",
            "Raw assistant capability remains quarantined.",
            "Structured/tool capability remains invalidated as model evidence.",
            "Not GPT-like readiness.",
            "Not open-domain assistant readiness.",
            "Not production chat.",
            "Not public API.",
            "Not deployment readiness.",
            "Not safety alignment.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def refresh_status(out: Path, status: str, verdicts: list[str], decision: dict[str, Any]) -> None:
    write_summary(out, status, verdicts, decision)
    write_report(out, verdicts, decision)


def verify_upstream_138g(out: Path, root: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_138G_ARTIFACTS if not (root / name).exists()]
    if missing:
        raise GateError("UPSTREAM_138G_ARTIFACT_MISSING", "138G artifacts missing", {"missing": missing})
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    scoring = read_json(root / "scoring_strictness_recheck.json")
    teacher = read_json(root / "teacher_forcing_vs_rollout_report.json")
    required = {
        "decision": decision.get("decision") == "objective_failure_ambiguous",
        "next": decision.get("next") == "138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION",
        "near_match_rate": isinstance(decision.get("near_match_rate"), (int, float)) and decision.get("near_match_rate") > 0.0,
        "expected_token_inclusion_rate": scoring.get("fields", {}).get("expected_token_inclusion_rate", {}).get("value") == 0.0,
        "teacher_forced_loss_initial_gap": teacher.get("fields", {}).get("teacher_forced_loss_initial", {}).get("evidence_type") == "diagnostic_gap",
        "teacher_forced_loss_final_gap": teacher.get("fields", {}).get("teacher_forced_loss_final", {}).get("evidence_type") == "diagnostic_gap",
        "artifact_only_analysis": decision.get("artifact_only_analysis") is True and summary.get("artifact_only_analysis") is True,
    }
    for key in FALSE_FLAGS:
        required[f"decision_{key}"] = decision.get(key) is False
        required[f"summary_{key}"] = summary.get(key) is False
    failed = [key for key, passed in required.items() if not passed]
    if failed:
        raise GateError("UPSTREAM_138G_ARTIFACT_MISSING", "138G did not match expected 138GA route", {"failed": failed})
    manifest = {
        "schema_version": "phase_138ga_upstream_138g_manifest_v1",
        "upstream_138g_root": rel(root),
        "upstream_138g_verified": True,
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "near_match_rate": decision.get("near_match_rate"),
        "expected_token_inclusion_rate": scoring["fields"]["expected_token_inclusion_rate"]["value"],
        "teacher_forced_loss_fields_diagnostic_gap": True,
        "artifact_only_analysis": True,
        **FALSE_FLAGS,
    }
    write_json(out / "upstream_138g_manifest.json", manifest)
    return manifest


def verify_upstream_138r(out: Path, root: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_138R_ARTIFACTS if not (root / name).exists()]
    if missing:
        raise GateError("UPSTREAM_138R_ARTIFACT_MISSING", "138R artifacts missing", {"missing": missing})
    aggregate = read_json(root / "aggregate_metrics.json")
    canary = read_json(root / "expected_output_canary_report.json")
    scan = read_json(root / "ast_shortcut_scan_report.json")
    controls = read_json(root / "control_arm_report.json")
    leakage = read_json(root / "freshness_leakage_audit.json")
    provenance = read_json(root / "helper_provenance_verification.json")
    generated_before = read_json(root / "generated_before_scoring_report.json")
    required = {
        "mean_accuracy_zero": aggregate.get("mean_real_raw_reasoning_accuracy") == 0.0,
        "expected_token_rate_zero": aggregate.get("expected_token_inclusion_rate") == 0.0,
        "near_match_nonzero": aggregate.get("near_match_rate", 0.0) > 0.0,
        "canary_passed": canary.get("expected_output_canary_passed") is True,
        "ast_passed": scan.get("ast_shortcut_scan_passed") is True,
        "controls_failed": controls.get("controls_failed") is True,
        "leakage_rejected": leakage.get("leakage_rejected") is True,
        "source_checkpoint_unchanged": provenance.get("source_checkpoint_unchanged") is True,
        "target_checkpoint_changed": provenance.get("target_checkpoint_changed") is True,
        "generated_before_scoring": generated_before.get("generated_text_produced_before_scoring") is True,
    }
    failed = [key for key, passed in required.items() if not passed]
    if failed:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138R helper or evidence integrity regression", {"failed": failed})
    manifest = {
        "schema_version": "phase_138ga_upstream_138r_manifest_v1",
        "upstream_138r_root": rel(root),
        "upstream_138r_verified": True,
        "row_count": aggregate.get("row_count"),
        "near_match_rate": aggregate.get("near_match_rate"),
        "expected_token_inclusion_rate": aggregate.get("expected_token_inclusion_rate"),
        "helper_canary_ast_leakage_controls_passed": True,
        "source_checkpoint_unchanged": True,
        "target_checkpoint_changed": True,
    }
    write_json(out / "upstream_138r_manifest.json", manifest)
    return manifest


def load_analysis_inputs(root_138g: Path, root_138r: Path) -> dict[str, Any]:
    return {
        "g_decision": read_json(root_138g / "decision.json"),
        "g_scoring": read_json(root_138g / "scoring_strictness_recheck.json"),
        "g_patterns": read_json(root_138g / "rollout_output_pattern_report.json"),
        "g_mismatch": read_json(root_138g / "first_mismatch_report.json"),
        "r_scoring": read_jsonl(root_138r / "scoring_results.jsonl"),
        "r_raw": read_jsonl(root_138r / "raw_generation_results.jsonl"),
        "r_eval": read_jsonl(root_138r / "eval_rows.jsonl"),
        "r_trace": read_jsonl(root_138r / "raw_generation_trace.jsonl"),
        "r_aggregate": read_json(root_138r / "aggregate_metrics.json"),
    }


def classify_near_matches(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_by_id = {row["row_id"]: row for row in inputs["r_raw"]}
    eval_by_id = {row["row_id"]: row for row in inputs["r_eval"]}
    trace_by_id = {row["row_id"]: row for row in inputs["r_trace"]}
    scored = inputs["r_scoring"]
    near_scores = [row for row in scored if row.get("near_match") is True]
    classified: list[dict[str, Any]] = []
    for score in sorted(near_scores, key=lambda row: row["row_id"]):
        row_id = score["row_id"]
        if row_id not in raw_by_id or row_id not in eval_by_id:
            raise GateError("NEAR_MATCH_CLASSIFICATION_INCOMPLETE", "near-match row missing raw/eval record", {"row_id": row_id})
        row = classify_row(score, raw_by_id[row_id], eval_by_id[row_id])
        trace = trace_by_id.get(row_id, {})
        row["helper_request_hash"] = trace.get("helper_request_hash")
        row["generation_trace_hash"] = raw_by_id[row_id].get("generation_trace_hash", trace.get("generation_trace_hash"))
        if row["primary_label"] not in PRIMARY_LABELS:
            raise GateError("NEAR_MATCH_CLASSIFICATION_INCOMPLETE", "invalid primary label", {"row_id": row_id, "label": row["primary_label"]})
        classified.append(row)
    total = len(scored)
    near_count = len(classified)
    rate = near_count / total if total else 0.0
    g_rate = inputs["g_decision"].get("near_match_rate")
    g_scoring_rate = inputs["g_scoring"].get("fields", {}).get("near_match_rate", {}).get("value")
    aggregate_rate = inputs["r_aggregate"].get("near_match_rate")
    rates = [item for item in [g_rate, g_scoring_rate, aggregate_rate] if isinstance(item, (int, float))]
    mismatches = [item for item in rates if abs(float(item) - rate) > 1e-12]
    if mismatches:
        raise GateError(
            "NEAR_MATCH_ARTIFACT_INCONSISTENCY",
            "computed near-match rate disagrees with upstream artifacts",
            {"computed_rate": rate, "upstream_rates": rates},
        )
    report = {
        "schema_version": "phase_138ga_near_match_extraction_v1",
        "total_scored_row_count": total,
        "near_match_row_count": near_count,
        "near_match_rate": rate,
        "upstream_rates_checked": rates,
        "near_match_rate_matches_138g": True,
        "source_artifacts": [
            "138R scoring_results.jsonl",
            "138R raw_generation_results.jsonl",
            "138R eval_rows.jsonl",
            "138R raw_generation_trace.jsonl",
            "138G scoring_strictness_recheck.json",
        ],
    }
    return classified, report


def build_reports(classified: list[dict[str, Any]], extraction: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    near_count = extraction["near_match_row_count"]
    total = extraction["total_scored_row_count"]
    counts = Counter(row["primary_label"] for row in classified)
    secondary_counts = Counter(label for row in classified for label in row.get("secondary_labels", []))
    meaningful_count = counts["meaningful_partial_answer"]
    formatting_count = counts["formatting_or_wrapper_mismatch"]
    unknown_count = counts["unknown_near_match"]
    trivial_count = sum(counts[label] for label in TRIVIAL_LABELS)
    nontrivial_count = sum(counts[label] for label in NONTRIVIAL_LABELS)
    meaningful_rate = meaningful_count / total if total else 0.0
    meaningful_fraction = meaningful_count / near_count if near_count else 0.0
    formatting_fraction_nontrivial = formatting_count / nontrivial_count if nontrivial_count else 0.0
    unknown_rate = unknown_count / near_count if near_count else 0.0
    trivial_rate = trivial_count / near_count if near_count else 0.0

    classification_report = {
        "schema_version": "phase_138ga_near_match_classification_report_v1",
        **extraction,
        "allowed_primary_labels": PRIMARY_LABELS,
        "exactly_one_primary_label_per_row": all(row["primary_label"] in PRIMARY_LABELS for row in classified),
        "primary_label_counts": dict(sorted(counts.items())),
        "secondary_label_counts": dict(sorted(secondary_counts.items())),
        "meaningful_near_match_rate": meaningful_rate,
        "meaningful_fraction_of_near_matches": meaningful_fraction,
        "formatting_or_wrapper_mismatch_rate_among_near_matches": formatting_count / near_count if near_count else 0.0,
        "formatting_or_wrapper_mismatch_rate_among_nontrivial": formatting_fraction_nontrivial,
        "unknown_near_match_rate": unknown_rate,
        "trivial_overlap_rate": trivial_rate,
        "deterministic_precedence_applied": [
            "exact expected token present",
            "train namespace ANSWER=T while expected ANSWER=E",
            "correct numeric answer token present",
            "expected answer body without wrapper",
            "stale User:/Assistant fragment dominates",
            "generated mostly copies prompt",
            "generated contains distractor token",
            "only common task token overlap",
            "weak scorer digit-substring overlap",
            "unknown",
        ],
    }
    meaningful_report = {
        "schema_version": "phase_138ga_meaningful_partial_answer_report_v1",
        "strict_criteria": [
            "exact expected output appears",
            "correct numeric answer appears as a standalone token and no train namespace mismatch is present",
            "expected answer body appears without wrapper and no train namespace mismatch is present",
            "explicit synonym or selected-option support if present in artifacts",
        ],
        "generic_overlap_rejected": True,
        "answer_prefix_only_rejected": True,
        "train_namespace_prefix_rejected": True,
        "prompt_copy_overlap_rejected": True,
        "stale_chat_text_rejected": True,
        "meaningful_near_match_count": meaningful_count,
        "meaningful_near_match_rate": meaningful_rate,
        "meaningful_fraction_of_near_matches": meaningful_fraction,
        "row_ids": [row["row_id"] for row in classified if row["primary_label"] == "meaningful_partial_answer"],
    }
    scorer_report = {
        "schema_version": "phase_138ga_scorer_eval_weakness_report_v1",
        "scorer_or_eval_design_contributes": meaningful_rate >= 0.02 or (formatting_count > 0 and formatting_fraction_nontrivial > 0.5),
        "meaningful_near_match_rate": meaningful_rate,
        "formatting_dominant_among_nontrivial": formatting_count > 0 and formatting_fraction_nontrivial > 0.5,
        "expected_token_inclusion_rate": 0.0,
        "near_match_source": "138R weak digit-substring near_match function",
        "classification_counts": dict(sorted(counts.items())),
    }
    disambiguation = {
        "schema_version": "phase_138ga_objective_failure_disambiguation_v1",
        "objective_failure_disambiguated": trivial_rate >= 0.8 and unknown_rate <= 0.10 and meaningful_rate < 0.02 and not scorer_report["formatting_dominant_among_nontrivial"],
        "mostly_trivial_overlap": trivial_rate >= 0.8,
        "unknown_bucket_material": unknown_rate > 0.10,
        "meaningful_signal_material": meaningful_rate >= 0.02,
        "formatting_or_wrapper_dominant": scorer_report["formatting_dominant_among_nontrivial"],
        "primary_label_counts": dict(sorted(counts.items())),
        "supporting_interpretation": "near matches are not capability evidence unless strict meaningful criteria pass",
    }
    decision_metrics = {
        "near_match_row_count": near_count,
        "total_scored_row_count": total,
        "near_match_rate": extraction["near_match_rate"],
        "meaningful_near_match_rate": meaningful_rate,
        "meaningful_fraction_of_near_matches": meaningful_fraction,
        "formatting_or_wrapper_mismatch_rate": formatting_count / near_count if near_count else 0.0,
        "formatting_or_wrapper_mismatch_rate_among_nontrivial": formatting_fraction_nontrivial,
        "unknown_near_match_rate": unknown_rate,
        "trivial_overlap_rate": trivial_rate,
        "primary_label_counts": dict(sorted(counts.items())),
    }
    return classification_report, meaningful_report, scorer_report, disambiguation, decision_metrics


def decide(metrics: dict[str, Any], disambiguation: dict[str, Any]) -> dict[str, Any]:
    if metrics["unknown_near_match_rate"] > 0.10:
        decision = "ambiguity_requires_manual_sample_review"
        next_step = "138GB_NEAR_MATCH_MANUAL_REVIEW_PACKET"
        verdict = "NEAR_MATCH_AMBIGUITY_REQUIRES_MANUAL_REVIEW"
    elif metrics["meaningful_near_match_rate"] >= 0.02 or disambiguation["formatting_or_wrapper_dominant"]:
        decision = "scorer_or_eval_design_contributes"
        next_step = "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"
        verdict = "SCORER_OR_EVAL_DESIGN_CONTRIBUTES"
    elif disambiguation["objective_failure_disambiguated"]:
        decision = "objective_failure_disambiguated"
        next_step = "138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN"
        verdict = "OBJECTIVE_FAILURE_DISAMBIGUATED"
    else:
        decision = "ambiguity_requires_manual_sample_review"
        next_step = "138GB_NEAR_MATCH_MANUAL_REVIEW_PACKET"
        verdict = "NEAR_MATCH_AMBIGUITY_REQUIRES_MANUAL_REVIEW"
    return {
        "schema_version": "phase_138ga_decision_v1",
        "verdict": verdict,
        "decision": decision,
        "next": next_step,
        "artifact_only_analysis": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "runtime_surface_mutated": False,
        "root_license_changed": False,
        **FALSE_FLAGS,
        **metrics,
    }


def write_failure(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if error.verdict == "UPSTREAM_138G_ARTIFACT_MISSING":
        decision_name = "upstream_138g_artifact_missing"
        next_name = "138GA_UPSTREAM_138G_ARTIFACT_MISSING"
    elif error.verdict == "UPSTREAM_138R_ARTIFACT_MISSING":
        decision_name = "upstream_138r_artifact_missing"
        next_name = "138GA_UPSTREAM_138R_ARTIFACT_MISSING"
    elif error.verdict == "NEAR_MATCH_ARTIFACT_INCONSISTENCY":
        decision_name = "near_match_artifact_inconsistency"
        next_name = "138GA_NEAR_MATCH_ARTIFACT_INCONSISTENCY_ANALYSIS"
    elif error.verdict == "NEAR_MATCH_CLASSIFICATION_INCOMPLETE":
        decision_name = "near_match_classification_incomplete"
        next_name = "138GB_NEAR_MATCH_MANUAL_REVIEW_PACKET"
    else:
        decision_name = "raw_helper_integrity_failure"
        next_name = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    decision = {
        "schema_version": "phase_138ga_failure_decision_v1",
        "verdict": error.verdict,
        "decision": decision_name,
        "next": next_name,
        "failure_message": error.message,
        "failure_details": error.details,
        "artifact_only_analysis": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [error.verdict], decision, error.message)
    write_report(out, [error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    root_138g = resolve_path(args.upstream_138g_root)
    root_138r = resolve_path(args.upstream_138r_root)
    write_json(
        out / "queue.json",
        {
            "schema_version": "phase_138ga_queue_v1",
            "milestone": MILESTONE,
            "status": "started",
            "started_at": utc_now(),
            "heartbeat_sec": args.heartbeat_sec,
        },
    )
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["138GA_RUNNING"], {"decision": "pending", "next": "pending"})

    verify_upstream_138g(out, root_138g)
    verify_upstream_138r(out, root_138r)
    append_progress(out, "upstream verification", upstream_138g_verified=True, upstream_138r_verified=True)
    refresh_status(out, "running", ["UPSTREAMS_VERIFIED"], {"decision": "pending", "next": "pending"})

    inputs = load_analysis_inputs(root_138g, root_138r)
    append_progress(out, "artifact loading", artifact_only=True)
    refresh_status(out, "running", ["ARTIFACTS_LOADED"], {"decision": "pending", "next": "pending"})

    classified, extraction = classify_near_matches(inputs)
    write_jsonl(out / "near_match_rows.jsonl", classified)
    append_progress(out, "near-match extraction", near_match_row_count=extraction["near_match_row_count"], near_match_rate=extraction["near_match_rate"])

    classification, meaningful, scorer, disambiguation, metrics = build_reports(classified, extraction)
    write_json(out / "near_match_classification_report.json", classification)
    write_json(out / "meaningful_partial_answer_report.json", meaningful)
    write_json(out / "scorer_eval_weakness_report.json", scorer)
    write_json(out / "objective_failure_disambiguation.json", disambiguation)
    write_jsonl(out / "human_readable_near_match_samples.jsonl", classified)
    append_progress(out, "near-match classification", primary_label_counts=metrics["primary_label_counts"])
    refresh_status(out, "running", ["NEAR_MATCH_ROWS_CLASSIFIED"], {"decision": "pending", "next": "pending", **metrics})

    decision = decide(metrics, disambiguation)
    write_json(out / "decision.json", decision)
    append_progress(out, "decision writing", decision=decision["decision"], next=decision["next"])

    verdicts = [
        decision["verdict"],
        "ARTIFACT_ONLY_NEAR_MATCH_DISAMBIGUATION",
        "NO_NEW_INFERENCE",
        "NO_HELPER_CALLS",
        "CAPABILITY_FLAGS_FALSE",
    ]
    refresh_status(out, "complete", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(
        out / "queue.json",
        {
            "schema_version": "phase_138ga_queue_v1",
            "milestone": MILESTONE,
            "status": "completed",
            "completed_at": utc_now(),
            "heartbeat_sec": args.heartbeat_sec,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138g-root", default=str(DEFAULT_UPSTREAM_138G_ROOT))
    parser.add_argument("--upstream-138r-root", default=str(DEFAULT_UPSTREAM_138R_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except GateError as exc:
        write_failure(args, exc)
        print(f"138GA failed closed: {exc.verdict}: {exc.message}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
