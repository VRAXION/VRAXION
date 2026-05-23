#!/usr/bin/env python3
"""138S artifact-only stale-chat rollout and value-grounding analysis.

This phase reads existing 138I artifacts only. It does not train, repair, run
new inference, call shared_raw_generation_helper.py, run torch forward passes,
mutate checkpoints, modify helper/backend code, import old runners, start
services, deploy, delete or consolidate files, or modify runtime/release/product
surfaces.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138s_stale_chat_rollout_failure_analysis/smoke")
DEFAULT_UPSTREAM_138I_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138i_real_raw_reasoning_rollout_aligned_repair_probe/smoke")
BOUNDARY_TEXT = (
    "138S is artifact-only stale-chat rollout failure analysis. It does not "
    "train, repair, run new inference, call shared_raw_generation_helper.py, "
    "run torch forward passes, mutate checkpoints, modify helper/backend code, "
    "import old runners, start services, deploy, delete or consolidate files, "
    "modify runtime/service/deploy/product/release surfaces, docs/product, "
    "docs/releases, or root LICENSE. It does not restore reasoning, raw "
    "assistant capability, structured/tool capability, GPT-like readiness, "
    "open-domain readiness, production chat, public API, deployment readiness, "
    "or safety alignment."
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
REQUIRED_138I_ARTIFACTS = [
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "namespace_metrics.json",
    "raw_generation_results.jsonl",
    "raw_generation_trace.jsonl",
    "scoring_results.jsonl",
    "eval_rows.jsonl",
    "train_rows.jsonl",
    "training_metrics.jsonl",
    "training_objective_report.json",
    "control_results.jsonl",
    "control_arm_report.json",
    "freshness_leakage_audit.json",
    "generated_before_scoring_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "source_checkpoint_integrity_manifest.json",
    "target_checkpoint_integrity_manifest.json",
    "determinism_replay_report.json",
    "failure_case_samples.jsonl",
    "human_readable_samples.jsonl",
]
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
FORBIDDEN_HELPER_KEYS = {
    "expected_output",
    "expected_payload",
    "expected_answer",
    "required_keys",
    "required_keywords",
    "forbidden_outputs",
    "schema_answer_object",
    "scorer_metadata",
    "labels",
    "oracle_data",
    "target_json",
    "gold_output",
    "row_answer",
    "eval_family",
    "answer",
    "expected_values",
}
PRIMARY_FAILURE_LABELS = [
    "stale_chat_with_wrong_value",
    "eval_namespace_but_wrong_value",
    "prefix_only_no_value",
    "off_prompt_without_stale",
    "empty_or_garbled_output",
    "train_namespace_regression",
    "correct_value_wrong_wrapper",
    "unknown_failure",
]
NEXT_OPTIONS = {
    "138T_STALE_CHAT_SUPPRESSION_AND_VALUE_GROUNDING_REPAIR_PLAN",
    "138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN",
    "138P_PROMPT_OUTPUT_FORMAT_REBALANCE_PLAN",
    "138Q_STALE_SOURCE_PRIOR_CHECKPOINT_REVIEW",
    "138E_SCORER_OR_TASK_WEAKNESS_ANALYSIS",
}


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


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
        raise GateError("138S_BOUNDARY_FAILURE", "--out must stay inside the repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138S_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    if any(part == ".." for part in relative.parts):
        raise GateError("138S_BOUNDARY_FAILURE", "--out must not escape target/pilot_wave")
    return resolved


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def answer_token(text: str) -> str | None:
    match = re.search(r"\bANSWER=[A-Za-z0-9_]+", text)
    return match.group(0) if match else None


def answer_body(token: str | None) -> str:
    if not token or "=" not in token:
        return ""
    return token.split("=", 1)[1]


def has_user(text: str) -> bool:
    return re.search(r"\bUser:", text) is not None


def has_assistant(text: str) -> bool:
    return re.search(r"\bAssistant:", text) is not None


def has_stale(text: str) -> bool:
    return has_user(text) or has_assistant(text)


def is_garbled(text: str) -> bool:
    stripped = text.strip()
    return not stripped or "\ufffd" in stripped


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138s_summary_v1",
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
            f"- `primary_diagnosis`: `{decision.get('primary_diagnosis')}`",
            f"- `stale_chat_fragment_rate`: `{decision.get('stale_chat_fragment_rate')}`",
            f"- `answer_value_accuracy`: `{decision.get('answer_value_accuracy')}`",
            f"- `prefix_success_value_failure_rate`: `{decision.get('prefix_success_value_failure_rate')}`",
            "",
            "138S is artifact-only analysis.",
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


def require_files(root: Path) -> None:
    missing = [name for name in REQUIRED_138I_ARTIFACTS if not (root / name).exists()]
    if missing:
        raise GateError("UPSTREAM_138I_ARTIFACT_MISSING", "required 138I artifacts missing", {"missing": missing})


def verify_upstream(out: Path, root: Path) -> dict[str, Any]:
    require_files(root)
    decision = read_json(root / "decision.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    namespace = read_json(root / "namespace_metrics.json")
    controls = read_json(root / "control_arm_report.json")
    leakage = read_json(root / "freshness_leakage_audit.json")
    before = read_json(root / "generated_before_scoring_report.json")
    canary = read_json(root / "expected_output_canary_report.json")
    scan = read_json(root / "ast_shortcut_scan_report.json")
    source = read_json(root / "source_checkpoint_integrity_manifest.json")
    target = read_json(root / "target_checkpoint_integrity_manifest.json")
    replay = read_json(root / "determinism_replay_report.json")
    traces = read_jsonl(root / "raw_generation_trace.jsonl")

    if decision.get("verdict") != "REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_FAILS" or decision.get("decision") != "stale_chat_rollout_failure":
        raise GateError("UPSTREAM_138I_ARTIFACT_MISSING", "138I did not route to stale-chat analysis")
    if decision.get("next") != "138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS":
        raise GateError("UPSTREAM_138I_ARTIFACT_MISSING", "138I next route is not 138S")
    if canary.get("expected_output_canary_passed") is not True or scan.get("ast_shortcut_scan_passed") is not True:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138I helper integrity gates did not pass")
    if controls.get("controls_failed") is not True or leakage.get("leakage_rejected") is not True:
        raise GateError("UPSTREAM_138I_ARTIFACT_MISSING", "138I controls or leakage gates did not pass")
    if replay.get("determinism_replay_passed") is not True or before.get("generated_text_produced_before_scoring") is not True:
        raise GateError("UPSTREAM_138I_ARTIFACT_MISSING", "138I determinism or generated-before-scoring gate did not pass")
    if source.get("source_checkpoint_unchanged") is not True or target.get("target_checkpoint_changed") is not True:
        raise GateError("UPSTREAM_138I_ARTIFACT_MISSING", "138I checkpoint integrity gates did not pass")
    if namespace.get("post_train_namespace_leak_rate") != 0.0:
        raise GateError("UPSTREAM_138I_ARTIFACT_MISSING", "138I train namespace leak profile changed")
    if namespace.get("post_eval_namespace_emission_accuracy") != 1.0 or namespace.get("post_answer_prefix_accuracy") != 1.0:
        raise GateError("UPSTREAM_138I_ARTIFACT_MISSING", "138I namespace/prefix profile changed")
    if namespace.get("post_answer_value_accuracy") != 0.0 or aggregate.get("mean_real_raw_reasoning_accuracy") != 0.0:
        raise GateError("UPSTREAM_138I_ARTIFACT_MISSING", "138I value/accuracy profile changed")
    if aggregate.get("stale_chat_fragment_rate", 0.0) <= 0.10:
        raise GateError("UPSTREAM_138I_ARTIFACT_MISSING", "138I stale rate is not above gate")
    for trace in traces:
        helper_request = trace.get("helper_request", {})
        if set(helper_request) != ALLOWED_HELPER_KEYS or set(helper_request) & FORBIDDEN_HELPER_KEYS:
            raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138I helper request contains unexpected metadata")
        if trace.get("generated_before_scoring") is not True:
            raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138I trace missing generated-before-scoring proof")

    manifest = {
        "schema_version": "phase_138s_upstream_138i_manifest_v1",
        "upstream_138i_root": rel(root),
        "upstream_138i_verified": True,
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "verdict": decision.get("verdict"),
        "post_train_namespace_leak_rate": namespace.get("post_train_namespace_leak_rate"),
        "post_eval_namespace_emission_accuracy": namespace.get("post_eval_namespace_emission_accuracy"),
        "post_answer_prefix_accuracy": namespace.get("post_answer_prefix_accuracy"),
        "post_answer_value_accuracy": namespace.get("post_answer_value_accuracy"),
        "mean_real_raw_reasoning_accuracy": aggregate.get("mean_real_raw_reasoning_accuracy"),
        "stale_chat_fragment_rate": aggregate.get("stale_chat_fragment_rate"),
        "helper_eval_integrity_passed": True,
        "no_expected_or_scorer_metadata_reached_helper_requests": True,
    }
    write_json(out / "upstream_138i_manifest.json", manifest)
    return manifest


def load_joined_rows(root: Path) -> list[dict[str, Any]]:
    raw_by_id = {row["row_id"]: row for row in read_jsonl(root / "raw_generation_results.jsonl")}
    eval_by_id = {row["row_id"]: row for row in read_jsonl(root / "eval_rows.jsonl")}
    joined: list[dict[str, Any]] = []
    for score in read_jsonl(root / "scoring_results.jsonl"):
        row_id = score["row_id"]
        raw = raw_by_id[row_id]
        eval_row = eval_by_id[row_id]
        text = raw["generated_text"]
        token = answer_token(text)
        body = answer_body(token)
        expected_body = answer_body(eval_row["expected_output"])
        stale_user = has_user(text)
        stale_assistant = has_assistant(text)
        item = {
            **score,
            "prompt": eval_row["prompt"],
            "generated_text": text,
            "expected_output": eval_row["expected_output"],
            "answer_token_computed": token,
            "answer_body": body,
            "expected_body": expected_body,
            "stale_user": stale_user,
            "stale_assistant": stale_assistant,
            "stale_chat": stale_user or stale_assistant,
            "wrong_value": not bool(score.get("answer_value_correct")),
            "empty_value": token is None or body in {"", "E"},
            "garbled_output": is_garbled(text),
        }
        joined.append(item)
    return joined


def stale_chat_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    stale_rows = [row for row in rows if row["stale_chat"]]
    stale_count = len(stale_rows)
    by_family: dict[str, dict[str, Any]] = {}
    by_seed: dict[str, dict[str, Any]] = {}
    for key_name, target in [("family", by_family), ("seed", by_seed)]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row[key_name])].append(row)
        for key, items in sorted(buckets.items()):
            count = sum(1 for item in items if item["stale_chat"])
            target[key] = {"row_count": len(items), "stale_chat_fragment_count": count, "stale_chat_fragment_rate": safe_rate(count, len(items))}
    return {
        "schema_version": "phase_138s_stale_chat_distribution_report_v1",
        "row_count": row_count,
        "stale_chat_fragment_count": stale_count,
        "stale_chat_fragment_rate": safe_rate(stale_count, row_count),
        "stale_user_rate": safe_rate(sum(1 for row in rows if row["stale_user"]), row_count),
        "stale_assistant_rate": safe_rate(sum(1 for row in rows if row["stale_assistant"]), row_count),
        "stale_both_rate": safe_rate(sum(1 for row in rows if row["stale_user"] and row["stale_assistant"]), row_count),
        "stale_by_family": by_family,
        "stale_by_seed": by_seed,
        "stale_with_answer_prefix_rate": safe_rate(sum(1 for row in stale_rows if row.get("answer_prefix_present")), stale_count),
        "stale_with_eval_namespace_rate": safe_rate(sum(1 for row in stale_rows if row.get("namespace_label") == "eval_namespace"), stale_count),
        "stale_with_wrong_value_rate": safe_rate(sum(1 for row in stale_rows if row["wrong_value"]), stale_count),
        "stale_without_answer_value_rate": safe_rate(sum(1 for row in stale_rows if not row.get("answer_value_correct")), stale_count),
    }


def value_grounding_failure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    prefix_count = sum(1 for row in rows if row.get("answer_prefix_present"))
    eval_ns_count = sum(1 for row in rows if row.get("namespace_label") == "eval_namespace")
    value_correct = sum(1 for row in rows if row.get("answer_value_correct"))
    exact_correct = sum(1 for row in rows if row.get("exact_answer_correct"))
    stale_wrong = sum(1 for row in rows if row["stale_chat"] and row["wrong_value"])
    no_stale_wrong = sum(1 for row in rows if not row["stale_chat"] and row["wrong_value"])
    return {
        "schema_version": "phase_138s_value_grounding_failure_report_v1",
        "row_count": row_count,
        "answer_prefix_accuracy": safe_rate(prefix_count, row_count),
        "eval_namespace_emission_accuracy": safe_rate(eval_ns_count, row_count),
        "answer_value_accuracy": safe_rate(value_correct, row_count),
        "exact_answer_accuracy": safe_rate(exact_correct, row_count),
        "rows_with_ANSWER_E_prefix": eval_ns_count,
        "rows_with_correct_value": value_correct,
        "rows_with_wrong_value": row_count - value_correct,
        "rows_with_empty_value": sum(1 for row in rows if row["empty_value"]),
        "rows_with_train_namespace": sum(1 for row in rows if row.get("namespace_label") == "train_namespace"),
        "rows_with_stale_fragment_and_wrong_value": stale_wrong,
        "rows_with_no_stale_fragment_but_wrong_value": no_stale_wrong,
        "value_failure_coupled_to_stale_fragments": stale_wrong > 0 and no_stale_wrong == 0,
        "value_grounding_fails_without_stale_fragments": no_stale_wrong > 0,
    }


def prefix_vs_value_decoupling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    prefix_only = sum(1 for row in rows if row.get("answer_prefix_present") and not row.get("answer_value_correct"))
    namespace_only = sum(1 for row in rows if row.get("namespace_label") == "eval_namespace" and not row.get("answer_value_correct"))
    value_success = sum(1 for row in rows if row.get("answer_value_correct"))
    return {
        "schema_version": "phase_138s_prefix_vs_value_decoupling_report_v1",
        "row_count": row_count,
        "prefix_success_value_failure_rate": safe_rate(prefix_only, row_count),
        "eval_namespace_success_value_failure_rate": safe_rate(namespace_only, row_count),
        "prefix_only_failure_count": prefix_only,
        "namespace_only_failure_count": namespace_only,
        "value_success_count": value_success,
        "wrapper_prefix_learned_without_value_grounding": prefix_only == row_count and value_success == 0,
    }


def source_prior_vs_training_objective(root: Path, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training_report = read_json(root / "training_objective_report.json")
    gaps: list[dict[str, Any]] = []
    def gap(field: str, note: str) -> dict[str, Any]:
        item = {"field": field, "evidence_type": "diagnostic_gap", "source": "138I artifacts", "note": note}
        gaps.append(item)
        return item

    stale_initial = training_report.get("stale_fragment_penalty_metric_initial")
    stale_final = training_report.get("stale_fragment_penalty_metric_final")
    report = {
        "schema_version": "phase_138s_source_prior_vs_training_objective_report_v1",
        "baseline_stale_fragment_rates": {
            "137R": gap("137R stale fragment rate", "not present in 138I artifact set"),
            "138R": gap("138R stale fragment rate", "not present in 138I artifact set"),
            "138I": read_json(root / "aggregate_metrics.json").get("stale_chat_fragment_rate"),
        },
        "stale_fragments_before_after_baseline_available": False,
        "training_objective": training_report.get("objective"),
        "training_loss_improved": training_report.get("training_loss_improved"),
        "training_objective_included_stale_penalty_field": "stale_fragment_penalty_metric_initial" in training_report and "stale_fragment_penalty_metric_final" in training_report,
        "stale_fragment_penalty_metric_initial": stale_initial,
        "stale_fragment_penalty_metric_final": stale_final,
        "stale_fragment_penalty_metric_changed": gap("stale fragment penalty metric changed", "metric values are null") if stale_initial is None or stale_final is None else stale_initial != stale_final,
        "rollout_alignment_metric_initial": training_report.get("rollout_alignment_metric_initial"),
        "rollout_alignment_metric_final": training_report.get("rollout_alignment_metric_final"),
        "train_rows_with_user_token": sum(1 for row in train_rows if "User:" in row["prompt"]),
        "train_rows_with_assistant_token": sum(1 for row in train_rows if "Assistant:" in row["prompt"]),
        "eval_rows_with_user_token": sum(1 for row in eval_rows if "User:" in row["prompt"]),
        "eval_rows_with_assistant_token": sum(1 for row in eval_rows if "Assistant:" in row["prompt"]),
        "train_expected_outputs_with_user_or_assistant": sum(1 for row in train_rows if "User:" in row["expected_output"] or "Assistant:" in row["expected_output"]),
        "eval_expected_outputs_with_user_or_assistant": sum(1 for row in eval_rows if "User:" in row["expected_output"] or "Assistant:" in row["expected_output"]),
        "source_checkpoint_prior_dominance": "diagnostic_gap",
        "source_checkpoint_prior_note": "138S does not run source-checkpoint inference; prior dominance cannot be proven from 138I artifacts alone.",
    }
    return report, gaps


def failure_label(row: dict[str, Any]) -> str:
    if row["stale_chat"] and row["wrong_value"]:
        return "stale_chat_with_wrong_value"
    if row.get("namespace_label") == "eval_namespace" and row["wrong_value"]:
        return "eval_namespace_but_wrong_value"
    if row.get("answer_prefix_present") and row["empty_value"]:
        return "prefix_only_no_value"
    if row.get("off_prompt_output") and not row["stale_chat"]:
        return "off_prompt_without_stale"
    if row["garbled_output"]:
        return "empty_or_garbled_output"
    if row.get("namespace_label") == "train_namespace" or row.get("train_namespace_leak"):
        return "train_namespace_regression"
    if row.get("answer_value_correct") and row.get("namespace_label") != "eval_namespace":
        return "correct_value_wrong_wrapper"
    return "unknown_failure"


def output_pattern_taxonomy(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    taxonomy_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("pass") is True:
            continue
        label = failure_label(row)
        counts[label] += 1
        taxonomy_rows.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "seed": row["seed"],
                "primary_failure_label": label,
                "generated_text": row["generated_text"],
                "expected_output": row["expected_output"],
                "answer_token": row["answer_token_computed"],
                "stale_chat": row["stale_chat"],
                "wrong_value": row["wrong_value"],
            }
        )
    report = {
        "schema_version": "phase_138s_output_pattern_taxonomy_v1",
        "failed_row_count": len(taxonomy_rows),
        "exactly_one_primary_label_per_failed_row": len(taxonomy_rows) == sum(counts.values()),
        "allowed_primary_failure_labels": PRIMARY_FAILURE_LABELS,
        "primary_failure_label_counts": dict(sorted(counts.items())),
        "unknown_failure_count": counts.get("unknown_failure", 0),
    }
    return report, taxonomy_rows


def stale_value_coupling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stale = [row for row in rows if row["stale_chat"]]
    no_stale = [row for row in rows if not row["stale_chat"]]
    wrong = [row for row in rows if row["wrong_value"]]
    correct = [row for row in rows if not row["wrong_value"]]
    p_wrong_stale = safe_rate(sum(1 for row in stale if row["wrong_value"]), len(stale))
    p_wrong_no_stale = safe_rate(sum(1 for row in no_stale if row["wrong_value"]), len(no_stale))
    return {
        "schema_version": "phase_138s_stale_chat_value_coupling_report_v1",
        "row_count": len(rows),
        "stale_chat_count": len(stale),
        "no_stale_chat_count": len(no_stale),
        "wrong_value_count": len(wrong),
        "correct_value_count": len(correct),
        "P_wrong_value_given_stale_chat": p_wrong_stale,
        "P_wrong_value_given_no_stale_chat": p_wrong_no_stale,
        "P_stale_chat_given_wrong_value": safe_rate(sum(1 for row in wrong if row["stale_chat"]), len(wrong)),
        "P_stale_chat_given_correct_value": safe_rate(sum(1 for row in correct if row["stale_chat"]), len(correct)),
        "stale_predictive_lift_for_wrong_value": None if p_wrong_stale is None or p_wrong_no_stale is None else p_wrong_stale - p_wrong_no_stale,
        "value_failure_occurs_without_stale_chat": sum(1 for row in no_stale if row["wrong_value"]) > 0,
        "stale_chat_is_sufficient_explanation_for_value_failure": len(no_stale) > 0 and sum(1 for row in no_stale if row["wrong_value"]) == 0,
    }


def choose_recommendation(value_report: dict[str, Any], prefix_report: dict[str, Any], coupling: dict[str, Any], stale_report: dict[str, Any]) -> dict[str, Any]:
    answer_value_accuracy = value_report["answer_value_accuracy"]
    stale_lift = coupling.get("stale_predictive_lift_for_wrong_value")
    if value_report["value_grounding_fails_without_stale_fragments"]:
        next_step = "138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN"
        reason = "wrong value persists even when stale chat fragments are absent"
        primary = "answer_value_grounding_failure_decoupled_from_stale_chat"
    elif stale_lift is not None and stale_lift > 0.25:
        next_step = "138T_STALE_CHAT_SUPPRESSION_AND_VALUE_GROUNDING_REPAIR_PLAN"
        reason = "stale chat strongly predicts wrong values"
        primary = "stale_chat_driven_value_failure"
    elif prefix_report["wrapper_prefix_learned_without_value_grounding"] or answer_value_accuracy == 0.0:
        next_step = "138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN"
        reason = "prefix and namespace succeeded while value accuracy remained zero"
        primary = "wrapper_success_without_value_grounding"
    elif stale_report["stale_chat_fragment_rate"] and stale_report["stale_chat_fragment_rate"] > 0.10:
        next_step = "138T_STALE_CHAT_SUPPRESSION_AND_VALUE_GROUNDING_REPAIR_PLAN"
        reason = "stale chat remains above gate"
        primary = "stale_chat_above_gate"
    else:
        next_step = "138SB_STALE_VALUE_MANUAL_REVIEW_PACKET"
        reason = "artifact-derived signals are ambiguous"
        primary = "stale_value_failure_ambiguous"
    return {
        "schema_version": "phase_138s_next_repair_recommendation_v1",
        "recommended_next": next_step,
        "primary_diagnosis": primary,
        "reason": reason,
        "allowed_options": sorted(NEXT_OPTIONS),
        "stale_chat_fragment_rate": stale_report["stale_chat_fragment_rate"],
        "answer_value_accuracy": answer_value_accuracy,
        "prefix_success_value_failure_rate": prefix_report["prefix_success_value_failure_rate"],
        "value_failure_occurs_without_stale_chat": coupling["value_failure_occurs_without_stale_chat"],
        "secondary_issue": "stale_chat_rollout_remains_above_gate" if stale_report["stale_chat_fragment_rate"] and stale_report["stale_chat_fragment_rate"] > 0.10 else None,
    }


def risk_register(recommendation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_138s_risk_register_v1",
        "risks": [
            {
                "risk": "future repair over-optimizes ANSWER=E wrapper without answer value grounding",
                "mitigation": "gate answer_value_accuracy and exact_answer_accuracy independently from prefix accuracy",
            },
            {
                "risk": "stale User:/Assistant fragments remain a co-occurring rollout defect",
                "mitigation": "keep stale fragment rate as a hard gate in the next repair",
            },
            {
                "risk": "source checkpoint prior cannot be isolated without new source-vs-target rollout",
                "mitigation": "treat source-prior dominance as diagnostic_gap until a dedicated artifact-safe review exists",
            },
        ],
        "recommended_next": recommendation["recommended_next"],
    }


def make_decision(recommendation: dict[str, Any], stale_report: dict[str, Any], value_report: dict[str, Any], prefix_report: dict[str, Any], coupling: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if recommendation["recommended_next"] == "138SB_STALE_VALUE_MANUAL_REVIEW_PACKET":
        decision_name = "stale_value_failure_ambiguous"
        next_step = "138SB_STALE_VALUE_MANUAL_REVIEW_PACKET"
        verdict = "STALE_VALUE_FAILURE_AMBIGUOUS"
    else:
        decision_name = "stale_chat_rollout_failure_analysis_complete"
        next_step = recommendation["recommended_next"]
        verdict = "STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS_COMPLETE"
    decision = {
        "schema_version": "phase_138s_decision_v1",
        "decision": decision_name,
        "next": next_step,
        "verdict": verdict,
        "primary_diagnosis": recommendation["primary_diagnosis"],
        "recommended_repair_type": recommendation["recommended_next"],
        "stale_chat_fragment_rate": stale_report["stale_chat_fragment_rate"],
        "stale_chat_fragment_count": stale_report["stale_chat_fragment_count"],
        "answer_prefix_accuracy": value_report["answer_prefix_accuracy"],
        "eval_namespace_emission_accuracy": value_report["eval_namespace_emission_accuracy"],
        "answer_value_accuracy": value_report["answer_value_accuracy"],
        "exact_answer_accuracy": value_report["exact_answer_accuracy"],
        "prefix_success_value_failure_rate": prefix_report["prefix_success_value_failure_rate"],
        "eval_namespace_success_value_failure_rate": prefix_report["eval_namespace_success_value_failure_rate"],
        "value_failure_occurs_without_stale_chat": coupling["value_failure_occurs_without_stale_chat"],
        "stale_chat_is_sufficient_explanation_for_value_failure": coupling["stale_chat_is_sufficient_explanation_for_value_failure"],
        "analysis_artifact_only": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "runtime_surface_mutated": False,
        "root_license_changed": False,
        **FALSE_FLAGS,
    }
    verdicts = [
        verdict,
        "ARTIFACT_ONLY_ANALYSIS",
        "UPSTREAM_138I_CLEAN_NEGATIVE_VERIFIED",
        "PREFIX_NAMESPACE_SUCCESS_VALUE_FAILURE_CONFIRMED",
        "STALE_CHAT_ABOVE_GATE_CONFIRMED",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED",
    ]
    return decision, verdicts


def write_failure_decision(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if error.verdict == "RAW_HELPER_INTEGRITY_FAILURE":
        decision_name = "raw_helper_integrity_failure"
        next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif error.verdict == "UPSTREAM_138I_ARTIFACT_MISSING":
        decision_name = "upstream_138i_artifact_missing"
        next_step = "138S_UPSTREAM_138I_ARTIFACT_MISSING"
    else:
        decision_name = "stale_value_failure_ambiguous"
        next_step = "138SB_STALE_VALUE_MANUAL_REVIEW_PACKET"
    decision = {
        "schema_version": "phase_138s_failure_decision_v1",
        "decision": decision_name,
        "next": next_step,
        "verdict": error.verdict,
        "failure_message": error.message,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [error.verdict], decision, error.message)
    write_report(out, [error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {"schema_version": "phase_138s_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS_RUNNING"], {"decision": "pending", "next": "pending"})

    upstream = resolve_path(args.upstream_138i_root)
    manifest = verify_upstream(out, upstream)
    append_progress(out, "upstream verification", upstream_138i_verified=True)
    refresh_status(out, "running", ["UPSTREAM_138I_VERIFIED"], {"decision": "pending", "next": "pending"})

    analysis_config = {
        "schema_version": "phase_138s_analysis_config_v1",
        "milestone": MILESTONE,
        "upstream_138i_root": rel(upstream),
        "artifact_only_analysis": True,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "heartbeat_sec": args.heartbeat_sec,
        "source_manifest": manifest,
    }
    write_json(out / "analysis_config.json", analysis_config)

    rows = load_joined_rows(upstream)
    train_rows = read_jsonl(upstream / "train_rows.jsonl")
    eval_rows = read_jsonl(upstream / "eval_rows.jsonl")
    append_progress(out, "artifact loading", scoring_rows=len(rows), train_rows=len(train_rows), eval_rows=len(eval_rows))

    stale_report = stale_chat_distribution(rows)
    write_json(out / "stale_chat_distribution_report.json", stale_report)
    append_progress(out, "stale distribution analysis", stale_chat_fragment_rate=stale_report["stale_chat_fragment_rate"])
    refresh_status(out, "running", ["STALE_DISTRIBUTION_ANALYZED"], {"decision": "pending", "next": "pending"})

    value_report = value_grounding_failure(rows)
    write_json(out / "value_grounding_failure_report.json", value_report)
    append_progress(out, "value grounding analysis", answer_value_accuracy=value_report["answer_value_accuracy"])

    prefix_report = prefix_vs_value_decoupling(rows)
    write_json(out / "prefix_vs_value_decoupling_report.json", prefix_report)
    append_progress(out, "prefix/value analysis", prefix_success_value_failure_rate=prefix_report["prefix_success_value_failure_rate"])

    source_report, source_gaps = source_prior_vs_training_objective(upstream, train_rows, eval_rows)
    write_json(out / "source_prior_vs_training_objective_report.json", source_report)
    append_progress(out, "source prior vs training objective analysis", diagnostic_gaps=len(source_gaps))

    taxonomy_report, taxonomy_rows = output_pattern_taxonomy(rows)
    write_json(out / "output_pattern_taxonomy.json", taxonomy_report)
    write_jsonl(out / "output_pattern_taxonomy_rows.jsonl", taxonomy_rows)
    append_progress(out, "taxonomy", failed_rows=taxonomy_report["failed_row_count"], unknown=taxonomy_report["unknown_failure_count"])

    coupling_report = stale_value_coupling(rows)
    write_json(out / "stale_chat_value_coupling_report.json", coupling_report)
    append_progress(out, "coupling analysis", value_failure_occurs_without_stale_chat=coupling_report["value_failure_occurs_without_stale_chat"])

    diagnostic_gaps = {
        "schema_version": "phase_138s_diagnostic_gap_register_v1",
        "gap_count": len(source_gaps),
        "gaps": source_gaps,
    }
    write_json(out / "diagnostic_gap_register.json", diagnostic_gaps)

    recommendation = choose_recommendation(value_report, prefix_report, coupling_report, stale_report)
    write_json(out / "next_repair_recommendation.json", recommendation)
    write_json(out / "risk_register.json", risk_register(recommendation))
    append_progress(out, "recommendation", next=recommendation["recommended_next"], primary_diagnosis=recommendation["primary_diagnosis"])

    decision, verdicts = make_decision(recommendation, stale_report, value_report, prefix_report, coupling_report)
    write_json(out / "decision.json", decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    refresh_status(out, "completed", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_138s_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138i-root", default=str(DEFAULT_UPSTREAM_138I_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except GateError as exc:
        write_failure_decision(args, exc)
        print(f"138S failed closed: {exc.verdict}: {exc.message}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
