#!/usr/bin/env python3
"""138G artifact-only real-raw reasoning objective failure analysis.

This phase reads existing 138R artifacts only. It does not train, run model
inference, call the shared helper, run torch forward passes, mutate checkpoints,
modify helper/backend code, import old runners, start services, or deploy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138G_REAL_RAW_REASONING_OBJECTIVE_FAILURE_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138g_real_raw_reasoning_objective_failure_analysis/smoke")
DEFAULT_UPSTREAM_138R_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138r_real_raw_reasoning_repair_training_plan_or_probe/smoke")
BOUNDARY_TEXT = (
    "138G is artifact-only analysis. It does not train, run new model "
    "inference, call shared_raw_generation_helper.py for new generations, run "
    "torch forward passes, mutate checkpoints, modify helper/backend code, "
    "import old runners, start services, deploy, delete or consolidate files, "
    "modify runtime/release/product surfaces, or change root LICENSE. It does "
    "not restore reasoning, raw assistant capability, structured/tool "
    "capability, GPT-like readiness, open-domain readiness, production chat, "
    "public API, deployment readiness, or safety alignment."
)
REQUIRED_138R_ARTIFACTS = [
    "decision.json",
    "aggregate_metrics.json",
    "training_metrics.jsonl",
    "raw_generation_results.jsonl",
    "raw_generation_trace.jsonl",
    "scoring_results.jsonl",
    "control_arm_report.json",
    "freshness_leakage_audit.json",
    "generated_before_scoring_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "helper_provenance_verification.json",
    "source_checkpoint_integrity_manifest.json",
    "target_checkpoint_integrity_manifest.json",
    "train_config.json",
    "eval_config.json",
    "train_rows.jsonl",
    "eval_rows.jsonl",
    "determinism_replay_report.json",
]
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
TAG_TYPES = {"artifact_observed", "computed_from_artifact", "diagnostic_gap", "inference"}


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
    if not path.exists():
        return []
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
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("138G_BOUNDARY_FAILURE", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138G_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def tagged(value: Any, evidence_type: str, source: str, note: str = "") -> dict[str, Any]:
    if evidence_type not in TAG_TYPES:
        raise ValueError(evidence_type)
    payload = {"value": value, "evidence_type": evidence_type, "source": source}
    if note:
        payload["note"] = note
    return payload


def gap(field: str, source: str, note: str) -> dict[str, Any]:
    return tagged(None, "diagnostic_gap", source, f"{field}: {note}")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138g_summary_v1",
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
            "shared_helper_called_for_new_generation": False,
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
            f"- `selected_root_cause`: `{decision.get('selected_root_cause')}`",
            f"- `near_match_rate`: `{decision.get('near_match_rate')}`",
            "",
            "138G is artifact-only analysis.",
            "Reasoning is not restored.",
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


def verify_upstream(out: Path, root: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_138R_ARTIFACTS if not (root / name).exists()]
    if missing:
        raise GateError("UPSTREAM_138R_ARTIFACT_MISSING", "138R artifacts missing", {"missing": missing})
    decision = read_json(root / "decision.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    canary = read_json(root / "expected_output_canary_report.json")
    scan = read_json(root / "ast_shortcut_scan_report.json")
    controls = read_json(root / "control_arm_report.json")
    leakage = read_json(root / "freshness_leakage_audit.json")
    provenance = read_json(root / "helper_provenance_verification.json")
    replay = read_json(root / "determinism_replay_report.json")
    required = {
        "verdict": decision.get("verdict") == "REAL_RAW_REASONING_REPAIR_PROBE_FAILS",
        "decision": decision.get("decision") == "teacher_forcing_or_training_objective_failure",
        "next": decision.get("next") == "138G_REAL_RAW_REASONING_OBJECTIVE_FAILURE_ANALYSIS",
        "determinism_replay_passed": decision.get("determinism_replay_passed") is True and replay.get("determinism_replay_passed") is True,
        "mean_real_raw_reasoning_accuracy": aggregate.get("mean_real_raw_reasoning_accuracy") == 0.0,
        "expected_token_inclusion_rate": aggregate.get("expected_token_inclusion_rate") == 0.0,
        "canary": canary.get("expected_output_canary_passed") is True,
        "ast": scan.get("ast_shortcut_scan_passed") is True,
        "controls": controls.get("controls_failed") is True,
        "leakage": leakage.get("leakage_rejected") is True,
        "source_checkpoint_unchanged": provenance.get("source_checkpoint_unchanged") is True,
        "target_checkpoint_changed": provenance.get("target_checkpoint_changed") is True,
    }
    failed = [key for key, passed in required.items() if not passed]
    if failed:
        raise GateError("UPSTREAM_138R_ARTIFACT_MISSING", "138R did not match expected 138G route", {"failed": failed})
    manifest = {
        "schema_version": "phase_138g_upstream_138r_manifest_v1",
        "upstream_138r_root": rel(root),
        "upstream_138r_verified": True,
        "verdict": decision.get("verdict"),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "determinism_replay_passed": True,
        "mean_real_raw_reasoning_accuracy": aggregate.get("mean_real_raw_reasoning_accuracy"),
        "expected_token_inclusion_rate": aggregate.get("expected_token_inclusion_rate"),
        "near_match_rate": aggregate.get("near_match_rate"),
        "helper_canary_ast_leakage_controls_passed": True,
        "source_checkpoint_unchanged": True,
        "target_checkpoint_changed": True,
    }
    write_json(out / "upstream_138r_manifest.json", manifest)
    return manifest


def load_artifacts(root: Path) -> dict[str, Any]:
    return {
        "decision": read_json(root / "decision.json"),
        "aggregate": read_json(root / "aggregate_metrics.json"),
        "training_metrics": read_jsonl(root / "training_metrics.jsonl"),
        "raw_results": read_jsonl(root / "raw_generation_results.jsonl"),
        "traces": read_jsonl(root / "raw_generation_trace.jsonl"),
        "scoring": read_jsonl(root / "scoring_results.jsonl"),
        "controls": read_json(root / "control_arm_report.json"),
        "leakage": read_json(root / "freshness_leakage_audit.json"),
        "generated_before": read_json(root / "generated_before_scoring_report.json"),
        "provenance": read_json(root / "helper_provenance_verification.json"),
        "source_manifest": read_json(root / "source_checkpoint_integrity_manifest.json"),
        "target_manifest": read_json(root / "target_checkpoint_integrity_manifest.json"),
        "train_config": read_json(root / "train_config.json"),
        "eval_config": read_json(root / "eval_config.json"),
        "train_rows": read_jsonl(root / "train_rows.jsonl"),
        "eval_rows": read_jsonl(root / "eval_rows.jsonl"),
        "replay": read_json(root / "determinism_replay_report.json"),
    }


def teacher_forcing_vs_rollout(root: Path, artifacts: dict[str, Any], gaps: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = artifacts["training_metrics"]
    aggregate = artifacts["aggregate"]
    first_loss = metrics[0].get("train_loss") if metrics else None
    final_loss = metrics[-1].get("train_loss") if metrics else None
    fields = {
        "train_loss_initial": tagged(first_loss, "artifact_observed", "training_metrics.jsonl[0].train_loss") if first_loss is not None else gap("train_loss_initial", "training_metrics.jsonl", "missing train_loss"),
        "train_loss_final": tagged(final_loss, "artifact_observed", "training_metrics.jsonl[-1].train_loss") if final_loss is not None else gap("train_loss_final", "training_metrics.jsonl", "missing train_loss"),
        "teacher_forced_loss_initial": gap("teacher_forced_loss_initial", "138R artifacts", "no dedicated teacher-forced initial loss field found"),
        "teacher_forced_loss_final": gap("teacher_forced_loss_final", "138R artifacts", "no dedicated teacher-forced final loss field found"),
        "rollout_final_eval_accuracy": tagged(aggregate.get("mean_real_raw_reasoning_accuracy"), "artifact_observed", "aggregate_metrics.json.mean_real_raw_reasoning_accuracy"),
        "expected_token_inclusion_rate": tagged(aggregate.get("expected_token_inclusion_rate"), "artifact_observed", "aggregate_metrics.json.expected_token_inclusion_rate"),
        "near_match_rate": tagged(aggregate.get("near_match_rate"), "artifact_observed", "aggregate_metrics.json.near_match_rate"),
    }
    for key, item in fields.items():
        if item["evidence_type"] == "diagnostic_gap":
            gaps.append({"field": key, "source": item["source"], "note": item["note"]})
    train_loss_moved = first_loss is not None and final_loss is not None and final_loss < first_loss
    return {
        "schema_version": "phase_138g_teacher_forcing_vs_rollout_report_v1",
        "artifact_root": rel(root),
        "evidence_tags_present": True,
        "fields": fields,
        "computed": {
            "train_loss_decreased": tagged(train_loss_moved, "computed_from_artifact", "training_metrics.jsonl first/last train_loss") if first_loss is not None and final_loss is not None else gap("train_loss_decreased", "training_metrics.jsonl", "cannot compute without first/final train_loss"),
            "teacher_forced_loss_improved_claim_allowed": tagged(False, "computed_from_artifact", "diagnostic gap policy", "dedicated teacher-forced loss fields are absent"),
            "loss_only_success_rejected": tagged(True, "computed_from_artifact", "aggregate_metrics.json + training_metrics.jsonl"),
        },
    }


def rollout_output_patterns(artifacts: dict[str, Any]) -> dict[str, Any]:
    raw = artifacts["raw_results"]
    scoring = {row["row_id"]: row for row in artifacts["scoring"]}
    eval_rows = {row["row_id"]: row for row in artifacts["eval_rows"]}
    row_count = len(raw)
    texts = [row.get("generated_text", "") for row in raw]
    hashes = [row.get("generated_text_hash") or text_hash(row.get("generated_text", "")) for row in raw]
    answer_prefix = sum(1 for text in texts if "ANSWER=" in text)
    train_ns = sum(1 for text in texts if re.search(r"ANSWER=T", text))
    eval_ns = sum(1 for text in texts if re.search(r"ANSWER=E", text))
    prompt_copy = 0
    for row in raw:
        prompt = eval_rows.get(row["row_id"], {}).get("prompt", "")
        text = row.get("generated_text", "")
        if prompt and (prompt in text or text in prompt):
            prompt_copy += 1
    metrics = {
        "row_count": tagged(row_count, "computed_from_artifact", "raw_generation_results.jsonl"),
        "unique_generated_text_hash_count": tagged(len(set(hashes)), "computed_from_artifact", "raw_generation_results.jsonl.generated_text_hash"),
        "repeated_output_rate": tagged(1.0 - (len(set(hashes)) / row_count if row_count else 0.0), "computed_from_artifact", "raw_generation_results.jsonl.generated_text_hash"),
        "stale_user_assistant_fragment_rate": tagged(sum(1 for text in texts if re.search(r"\\b(User|Assistant):", text)) / row_count if row_count else 0.0, "computed_from_artifact", "raw_generation_results.jsonl.generated_text"),
        "answer_prefix_rate": tagged(answer_prefix / row_count if row_count else 0.0, "computed_from_artifact", "raw_generation_results.jsonl.generated_text"),
        "expected_token_inclusion_rate": tagged(sum(1 for row in scoring.values() if row.get("expected_token_included")) / row_count if row_count else 0.0, "computed_from_artifact", "scoring_results.jsonl.expected_token_included"),
        "train_namespace_token_rate": tagged(train_ns / row_count if row_count else 0.0, "computed_from_artifact", "raw_generation_results.jsonl.generated_text"),
        "eval_namespace_token_rate": tagged(eval_ns / row_count if row_count else 0.0, "computed_from_artifact", "raw_generation_results.jsonl.generated_text"),
        "off_prompt_output_rate": tagged(sum(1 for row in scoring.values() if row.get("off_prompt_output")) / row_count if row_count else 0.0, "computed_from_artifact", "scoring_results.jsonl.off_prompt_output"),
        "empty_output_rate": tagged(sum(1 for text in texts if not text.strip()) / row_count if row_count else 0.0, "computed_from_artifact", "raw_generation_results.jsonl.generated_text"),
        "utf8_replacement_rate": tagged(sum(1 for text in texts if "\ufffd" in text) / row_count if row_count else 0.0, "computed_from_artifact", "raw_generation_results.jsonl.generated_text"),
        "prompt_copy_rate": tagged(prompt_copy / row_count if row_count else 0.0, "computed_from_artifact", "raw_generation_results.jsonl + eval_rows.jsonl"),
    }
    return {"schema_version": "phase_138g_rollout_output_pattern_report_v1", "evidence_tags_present": True, "metrics": metrics}


def namespace_report(artifacts: dict[str, Any], gaps: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = artifacts["train_rows"]
    eval_rows = artifacts["eval_rows"]
    raw = artifacts["raw_results"]
    train_has = any(re.search(r"ANSWER=T", row.get("expected_output", "")) for row in train_rows)
    eval_has = any(re.search(r"ANSWER=E", row.get("expected_output", "")) for row in eval_rows)
    generated_train = sum(1 for row in raw if re.search(r"ANSWER=T", row.get("generated_text", "")))
    generated_eval = sum(1 for row in raw if re.search(r"ANSWER=E", row.get("generated_text", "")))
    if not train_has:
        gaps.append({"field": "train_answer_namespace", "source": "train_rows.jsonl", "note": "ANSWER=T namespace absent"})
    if not eval_has:
        gaps.append({"field": "eval_answer_namespace", "source": "eval_rows.jsonl", "note": "ANSWER=E namespace absent"})
    row_count = len(raw)
    return {
        "schema_version": "phase_138g_train_eval_answer_namespace_report_v1",
        "evidence_tags_present": True,
        "fields": {
            "train_namespace_present": tagged(train_has, "computed_from_artifact", "train_rows.jsonl.expected_output"),
            "eval_namespace_present": tagged(eval_has, "computed_from_artifact", "eval_rows.jsonl.expected_output"),
            "generated_train_namespace_token_count": tagged(generated_train, "computed_from_artifact", "raw_generation_results.jsonl.generated_text"),
            "generated_eval_namespace_token_count": tagged(generated_eval, "computed_from_artifact", "raw_generation_results.jsonl.generated_text"),
            "generated_train_namespace_token_rate": tagged(generated_train / row_count if row_count else 0.0, "computed_from_artifact", "raw_generation_results.jsonl.generated_text"),
            "generated_eval_namespace_token_rate": tagged(generated_eval / row_count if row_count else 0.0, "computed_from_artifact", "raw_generation_results.jsonl.generated_text"),
        },
    }


def first_mismatch_and_alignment(artifacts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_by_id = {row["row_id"]: row for row in artifacts["raw_results"]}
    eval_by_id = {row["row_id"]: row for row in artifacts["eval_rows"]}
    rows: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    for score in artifacts["scoring"]:
        row_id = score["row_id"]
        generated = raw_by_id.get(row_id, {}).get("generated_text", "")
        expected = score.get("expected_output", eval_by_id.get(row_id, {}).get("expected_output", ""))
        gen = generated.lstrip()
        exp = str(expected).lstrip()
        first_match = bool(gen and exp and gen[0] == exp[0])
        expected_prefix = exp.split("=")[0] + "=" if "=" in exp else exp[: min(6, len(exp))]
        answer_marker_present = "ANSWER=" in generated
        position = generated.find(exp)
        if not gen:
            category = "empty_output"
        elif position >= 0:
            category = "exact_expected_present"
        elif "ANSWER=T" in generated and "ANSWER=E" in exp:
            category = "train_namespace_instead_of_eval_namespace"
        elif answer_marker_present:
            category = "answer_marker_wrong_value"
        elif re.search(r"\b(User|Assistant):", generated):
            category = "stale_fragment"
        else:
            category = "missing_answer_marker"
        categories[category] += 1
        if len(rows) < 100:
            rows.append(
                {
                    "row_id": row_id,
                    "first_nonwhitespace_char_match": first_match,
                    "expected_prefix_present": expected_prefix in generated,
                    "answer_marker_present": answer_marker_present,
                    "first_expected_token_position": position,
                    "mismatch_category": category,
                    "expected_output": expected,
                    "generated_text_sample": generated[:160],
                    "evidence_type": "computed_from_artifact",
                    "source": "raw_generation_results.jsonl + scoring_results.jsonl",
                }
            )
    total = len(artifacts["scoring"])
    mismatch_report = {
        "schema_version": "phase_138g_first_mismatch_report_v1",
        "evidence_tags_present": True,
        "row_count": tagged(total, "computed_from_artifact", "scoring_results.jsonl"),
        "mismatch_category_counts": tagged(dict(sorted(categories.items())), "computed_from_artifact", "raw_generation_results.jsonl + scoring_results.jsonl"),
        "sample_rows": rows,
    }
    align = {
        "schema_version": "phase_138g_prompt_answer_alignment_report_v1",
        "evidence_tags_present": True,
        "fields": {
            "answer_marker_present_rate": tagged(sum(1 for score in artifacts["scoring"] if "ANSWER=" in raw_by_id.get(score["row_id"], {}).get("generated_text", "")) / total if total else 0.0, "computed_from_artifact", "raw_generation_results.jsonl.generated_text"),
            "exact_expected_present_rate": tagged(sum(1 for score in artifacts["scoring"] if str(score.get("expected_output", "")) in raw_by_id.get(score["row_id"], {}).get("generated_text", "")) / total if total else 0.0, "computed_from_artifact", "raw_generation_results.jsonl + scoring_results.jsonl"),
            "dominant_mismatch_category": tagged(categories.most_common(1)[0][0] if categories else None, "computed_from_artifact", "first_mismatch_report category counts"),
        },
    }
    return mismatch_report, align


def stop_behavior(artifacts: dict[str, Any]) -> dict[str, Any]:
    traces = artifacts["traces"]
    raw = artifacts["raw_results"]
    stop_reasons = Counter((trace.get("response") or {}).get("stop_reason") for trace in traces)
    token_counts = [int(row.get("token_count", 0)) for row in raw]
    row_count = len(raw)
    return {
        "schema_version": "phase_138g_stop_behavior_report_v1",
        "evidence_tags_present": True,
        "fields": {
            "stop_reason_counts": tagged(dict(sorted(stop_reasons.items())), "computed_from_artifact", "raw_generation_trace.jsonl.response.stop_reason"),
            "token_count_min": tagged(min(token_counts) if token_counts else None, "computed_from_artifact", "raw_generation_results.jsonl.token_count"),
            "token_count_mean": tagged(mean(token_counts) if token_counts else None, "computed_from_artifact", "raw_generation_results.jsonl.token_count"),
            "token_count_max": tagged(max(token_counts) if token_counts else None, "computed_from_artifact", "raw_generation_results.jsonl.token_count"),
            "max_new_tokens_stop_rate": tagged(stop_reasons.get("max_new_tokens", 0) / row_count if row_count else 0.0, "computed_from_artifact", "raw_generation_trace.jsonl.response.stop_reason"),
            "explicit_eos_observed": tagged(any(reason not in {None, "max_new_tokens"} for reason in stop_reasons), "computed_from_artifact", "raw_generation_trace.jsonl.response.stop_reason"),
        },
    }


def scoring_recheck(artifacts: dict[str, Any]) -> dict[str, Any]:
    aggregate = artifacts["aggregate"]
    controls_failed = artifacts["controls"].get("controls_failed") is True
    leakage_rejected = artifacts["leakage"].get("leakage_rejected") is True
    expected_rate = aggregate.get("expected_token_inclusion_rate")
    near_rate = aggregate.get("near_match_rate")
    if expected_rate == 0.0 and near_rate == 0.0 and controls_failed and leakage_rejected:
        classification = "scoring_strict_but_valid"
    elif near_rate is None:
        classification = "diagnostic_gap"
    else:
        classification = "scoring_or_task_weakness_possible"
    return {
        "schema_version": "phase_138g_scoring_strictness_recheck_v1",
        "evidence_tags_present": True,
        "classification": tagged(classification, "computed_from_artifact", "aggregate_metrics.json + control_arm_report.json + freshness_leakage_audit.json"),
        "fields": {
            "expected_token_inclusion_rate": tagged(expected_rate, "artifact_observed", "aggregate_metrics.json.expected_token_inclusion_rate"),
            "near_match_rate": tagged(near_rate, "artifact_observed", "aggregate_metrics.json.near_match_rate"),
            "controls_failed": tagged(controls_failed, "artifact_observed", "control_arm_report.json.controls_failed"),
            "leakage_rejected": tagged(leakage_rejected, "artifact_observed", "freshness_leakage_audit.json.leakage_rejected"),
            "near_match_nonzero_requires_ambiguity_route": tagged(bool(near_rate and near_rate > 0.0), "computed_from_artifact", "aggregate_metrics.json.near_match_rate"),
        },
    }


def checkpoint_gap_report(artifacts: dict[str, Any], teacher_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_138g_checkpoint_objective_gap_report_v1",
        "evidence_tags_present": True,
        "fields": {
            "source_checkpoint_unchanged": tagged(artifacts["provenance"].get("source_checkpoint_unchanged"), "artifact_observed", "helper_provenance_verification.json.source_checkpoint_unchanged"),
            "target_checkpoint_changed": tagged(artifacts["provenance"].get("target_checkpoint_changed"), "artifact_observed", "helper_provenance_verification.json.target_checkpoint_changed"),
            "target_helper_load_status": tagged(artifacts["provenance"].get("backend_load_status"), "artifact_observed", "helper_provenance_verification.json.backend_load_status"),
            "train_loss_decreased": teacher_report["computed"]["train_loss_decreased"],
            "rollout_accuracy": tagged(artifacts["aggregate"].get("mean_real_raw_reasoning_accuracy"), "artifact_observed", "aggregate_metrics.json.mean_real_raw_reasoning_accuracy"),
            "target_changed_without_rollout_success": tagged(
                artifacts["provenance"].get("target_checkpoint_changed") is True and artifacts["aggregate"].get("mean_real_raw_reasoning_accuracy") == 0.0,
                "computed_from_artifact",
                "helper_provenance_verification.json + aggregate_metrics.json",
            ),
        },
    }


def root_cause(artifacts: dict[str, Any], scoring: dict[str, Any], namespace: dict[str, Any], gaps: list[dict[str, Any]]) -> dict[str, Any]:
    near_rate = artifacts["aggregate"].get("near_match_rate")
    selected = "objective_failure_ambiguous" if near_rate and near_rate > 0.0 else "rollout_objective_mismatch"
    support = [
        {"claim": "final rollout accuracy remained zero", "evidence_type": "artifact_observed", "source": "aggregate_metrics.json.mean_real_raw_reasoning_accuracy", "value": artifacts["aggregate"].get("mean_real_raw_reasoning_accuracy")},
        {"claim": "expected token inclusion remained zero", "evidence_type": "artifact_observed", "source": "aggregate_metrics.json.expected_token_inclusion_rate", "value": artifacts["aggregate"].get("expected_token_inclusion_rate")},
        {"claim": "near-match nonzero forces ambiguity route", "evidence_type": "computed_from_artifact", "source": "aggregate_metrics.json.near_match_rate", "value": bool(near_rate and near_rate > 0.0)},
        {"claim": "generated rollout prefers train namespace tokens", "evidence_type": "computed_from_artifact", "source": "train_eval_answer_namespace_report.json", "value": namespace["fields"]["generated_train_namespace_token_rate"]["value"]},
    ]
    return {
        "schema_version": "phase_138g_objective_failure_root_cause_v1",
        "evidence_tags_present": True,
        "allowed_root_causes": [
            "rollout_objective_mismatch",
            "teacher_forcing_only_failure",
            "answer_format_alignment_failure",
            "stop_or_eos_behavior_failure",
            "prompt_distribution_mismatch",
            "checkpoint_capacity_or_initialization_gap",
            "objective_failure_ambiguous",
        ],
        "selected_root_cause": tagged(selected, "computed_from_artifact", "aggregate/scoring/namespace reports"),
        "support": support,
        "diagnostic_gap_count": tagged(len(gaps), "computed_from_artifact", "diagnostic_gap_register.json"),
        "overclaim_rejected": tagged(True, "computed_from_artifact", "root-cause conservative routing policy"),
    }


def next_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_138g_next_objective_redesign_requirements_v1",
        "milestone": "138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN",
        "blocked_until_ambiguity_resolved_if_needed": True,
        "must_address_rollout_alignment_directly": True,
        "required_gates": [
            "helper-only final eval",
            "generated_text before scoring",
            "expected-output canary",
            "AST shortcut scan",
            "deterministic replay",
            "controls fail",
            "leakage rejected",
            "clean negative accepted",
        ],
        "explicitly_reject": [
            "teacher-forcing-only success",
            "loss-only success",
            "expected-output construction",
            "old runner imports",
            "oracle/rerank/verifier/LLM judge",
            "post-generation repair",
            "threshold weakening to force positive",
        ],
    }


def decide(root_cause_report: dict[str, Any], scoring: dict[str, Any]) -> dict[str, Any]:
    selected = root_cause_report["selected_root_cause"]["value"]
    near_nonzero = scoring["fields"]["near_match_nonzero_requires_ambiguity_route"]["value"]
    if near_nonzero or selected == "objective_failure_ambiguous":
        decision_name = "objective_failure_ambiguous"
        next_name = "138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION"
    else:
        decision_name = "objective_failure_analysis_complete"
        next_name = "138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN"
    return {
        "schema_version": "phase_138g_decision_v1",
        "decision": decision_name,
        "next": next_name,
        "selected_root_cause": selected,
        "near_match_rate": scoring["fields"]["near_match_rate"]["value"],
        "scoring_classification": scoring["classification"]["value"],
        "artifact_only_analysis": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called_for_new_generation": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "runtime_surface_mutated": False,
        "root_license_changed": False,
        **FALSE_FLAGS,
    }


def write_failure(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if error.verdict == "UPSTREAM_138R_ARTIFACT_MISSING":
        decision_name = "upstream_138r_artifact_missing"
        next_name = "138G_UPSTREAM_138R_ARTIFACT_MISSING"
    else:
        decision_name = "raw_helper_integrity_failure"
        next_name = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    decision = {
        "schema_version": "phase_138g_failure_decision_v1",
        "decision": decision_name,
        "next": next_name,
        "failure_verdict": error.verdict,
        "failure_message": error.message,
        "artifact_only_analysis": True,
        "training_performed": False,
        "new_model_inference_run": False,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [error.verdict], decision, error.message)
    write_report(out, [error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    root = resolve_path(args.upstream_138r_root)
    write_json(out / "queue.json", {"schema_version": "phase_138g_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["OBJECTIVE_FAILURE_ANALYSIS_RUNNING"], {"decision": "pending", "next": "pending"})

    manifest = verify_upstream(out, root)
    append_progress(out, "upstream verification", upstream_138r_verified=True)
    refresh_status(out, "running", ["UPSTREAM_138R_VERIFIED"], {"decision": "pending", "next": "pending"})

    artifacts = load_artifacts(root)
    append_progress(out, "artifact loading", artifact_count=len(REQUIRED_138R_ARTIFACTS))
    config = {
        "schema_version": "phase_138g_objective_failure_config_v1",
        "artifact_only_analysis": True,
        "upstream_138r_root": rel(root),
        "field_tag_types": sorted(TAG_TYPES),
        "no_training": True,
        "no_new_inference": True,
        "no_helper_calls_for_new_generations": True,
        "no_torch_forward_passes": True,
        "heartbeat_sec": args.heartbeat_sec,
    }
    write_json(out / "objective_failure_config.json", config)

    gaps: list[dict[str, Any]] = []
    teacher = teacher_forcing_vs_rollout(root, artifacts, gaps)
    write_json(out / "teacher_forcing_vs_rollout_report.json", teacher)
    append_progress(out, "teacher forcing vs rollout analysis", diagnostic_gaps=len(gaps))

    patterns = rollout_output_patterns(artifacts)
    write_json(out / "rollout_output_pattern_report.json", patterns)
    append_progress(out, "rollout output pattern analysis", row_count=patterns["metrics"]["row_count"]["value"])

    namespace = namespace_report(artifacts, gaps)
    write_json(out / "train_eval_answer_namespace_report.json", namespace)
    append_progress(out, "namespace analysis", train_namespace_rate=namespace["fields"]["generated_train_namespace_token_rate"]["value"])

    mismatch, alignment = first_mismatch_and_alignment(artifacts)
    write_json(out / "first_mismatch_report.json", mismatch)
    write_json(out / "prompt_answer_alignment_report.json", alignment)
    append_progress(out, "first mismatch and prompt alignment analysis")

    stop = stop_behavior(artifacts)
    write_json(out / "stop_behavior_report.json", stop)
    append_progress(out, "stop behavior analysis", stop_reasons=stop["fields"]["stop_reason_counts"]["value"])

    scoring = scoring_recheck(artifacts)
    write_json(out / "scoring_strictness_recheck.json", scoring)
    append_progress(out, "scoring strictness recheck", classification=scoring["classification"]["value"])

    checkpoint_gap = checkpoint_gap_report(artifacts, teacher)
    write_json(out / "checkpoint_objective_gap_report.json", checkpoint_gap)
    append_progress(out, "checkpoint objective gap analysis")

    write_json(out / "diagnostic_gap_register.json", {"schema_version": "phase_138g_diagnostic_gap_register_v1", "diagnostic_gap_count": len(gaps), "gaps": gaps})
    root_cause_report = root_cause(artifacts, scoring, namespace, gaps)
    write_json(out / "objective_failure_root_cause.json", root_cause_report)
    append_progress(out, "root-cause selection", selected=root_cause_report["selected_root_cause"]["value"])

    requirements = next_requirements()
    write_json(out / "next_objective_redesign_requirements.json", requirements)
    write_json(
        out / "risk_register.json",
        {
            "schema_version": "phase_138g_risk_register_v1",
            "risks": [
                {"risk": "near-match nonzero may indicate scorer/eval ambiguity", "mitigation": "route to 138GA before new training if present"},
                {"risk": "teacher-forced loss fields missing", "mitigation": "record diagnostic_gap and reject unsupported claims"},
                {"risk": "loss-only repair repeats 138R failure", "mitigation": "require rollout-aligned objective redesign"},
            ],
        },
    )
    append_progress(out, "next-plan drafting", next=requirements["milestone"])

    decision = decide(root_cause_report, scoring)
    write_json(out / "decision.json", decision)
    append_progress(out, "decision writing", decision=decision["decision"], next=decision["next"])
    verdicts = [
        "OBJECTIVE_FAILURE_ANALYSIS_COMPLETE" if decision["decision"] == "objective_failure_analysis_complete" else "OBJECTIVE_FAILURE_AMBIGUOUS",
        "ARTIFACT_ONLY_ANALYSIS",
        "NO_NEW_INFERENCE",
        "NO_HELPER_CALLS_FOR_NEW_GENERATION",
        "DIAGNOSTIC_GAPS_RECORDED",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED",
    ]
    refresh_status(out, "complete", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_138g_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
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
        print(f"138G failed closed: {exc.verdict}: {exc.message}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
