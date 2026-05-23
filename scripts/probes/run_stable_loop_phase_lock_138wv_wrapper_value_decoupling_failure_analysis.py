#!/usr/bin/env python3
"""138WV artifact-only wrapper/value decoupling failure analysis.

This phase reads existing 138W artifacts only. It does not train, repair, run
new inference, call shared_raw_generation_helper.py, run torch forward passes,
mutate checkpoints, modify helper/backend code, import old runners, delete or
consolidate files, start services, deploy, or modify runtime/release/product
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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138WV_WRAPPER_VALUE_DECOUPLING_FAILURE_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138wv_wrapper_value_decoupling_failure_analysis/smoke")
DEFAULT_UPSTREAM_138W_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138w_answer_value_grounding_repair_probe/smoke")
BOUNDARY_TEXT = (
    "138WV is artifact-only analysis. It reads existing 138W artifacts only and "
    "does not train, repair, run new inference, call shared_raw_generation_helper.py, "
    "run torch forward passes, mutate checkpoints, modify helper/backend code, "
    "import old runners, delete or consolidate files, start services, deploy, "
    "modify runtime/service/deploy/product/release surfaces, modify SDK exports, "
    "modify docs/product or docs/releases, or change root LICENSE."
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
REQUIRED_138W_ARTIFACTS = [
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "value_grounding_metrics.json",
    "parrot_trap_report.json",
    "post_wrapper_carrier_proxy_report.json",
    "raw_generation_results.jsonl",
    "raw_generation_trace.jsonl",
    "scoring_results.jsonl",
    "eval_rows.jsonl",
    "control_results.jsonl",
    "control_arm_report.json",
    "freshness_leakage_audit.json",
    "generated_before_scoring_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "determinism_replay_report.json",
    "source_checkpoint_integrity_manifest.json",
    "target_checkpoint_integrity_manifest.json",
    "failure_case_samples.jsonl",
    "human_readable_samples.jsonl",
]
PRIMARY_LABELS = [
    "immediate_termination_proxy",
    "empty_or_whitespace_after_wrapper",
    "default_neutral_attractor",
    "structural_format_echo",
    "generic_wrong_value",
    "repeated_symbol_or_punctuation",
    "wrong_specific_value",
    "delayed_correct_value_wrong_position",
    "garbled_after_wrapper",
    "unknown_post_wrapper_behavior",
]
NEUTRAL_DEFAULTS = {"0", "none", "null", "unknown", "n/a", "false", "empty", "na", "nil"}
STRUCTURAL_TOKENS = {"question", "prompt", "fact", "row", "user", "assistant", "answer", "case", "sample", "next"}
GENERIC_VALUES = {"value", "token", "answer", "unknown", "result", "number", "item", "data"}


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
        raise GateError("138WV_BOUNDARY_FAILURE", "--out must stay inside repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138WV_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    return resolved


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138wv_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "artifact_only": True,
            "new_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            "literal_eos_claimed": False,
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
        "## EOS Guardrail",
        "",
        "Immediate termination is measured only as `immediate_termination_proxy_rate`. It is not a literal EOS-token claim because the shared helper records `stop_reason = max_new_tokens`.",
        "Topological inhibition language is hypothesis/inference only unless future instrumentation supports it.",
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
            f"- `root_cause`: `{decision.get('root_cause')}`",
            f"- `literal_eos_claimed`: `{decision.get('literal_eos_claimed')}`",
            "",
            "Reasoning is not restored.",
            "Raw assistant capability remains quarantined.",
            "Structured/tool capability remains invalidated.",
            "not GPT-like readiness.",
            "not open-domain assistant readiness.",
            "not production chat.",
            "not public API.",
            "not deployment readiness.",
            "not safety alignment.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def refresh_status(out: Path, status: str, verdicts: list[str], decision: dict[str, Any]) -> None:
    write_summary(out, status, verdicts, decision)
    write_report(out, verdicts, decision)


def verify_upstream(out: Path, root_138w: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_138W_ARTIFACTS if not (root_138w / name).exists()]
    if missing:
        raise GateError("UPSTREAM_138W_ARTIFACT_MISSING", "required 138W artifacts missing", {"missing": missing})
    decision = read_json(root_138w / "decision.json")
    aggregate = read_json(root_138w / "aggregate_metrics.json")
    value = read_json(root_138w / "value_grounding_metrics.json")
    parrot = read_json(root_138w / "parrot_trap_report.json")
    carrier = read_json(root_138w / "post_wrapper_carrier_proxy_report.json")
    controls = read_json(root_138w / "control_arm_report.json")
    leakage = read_json(root_138w / "freshness_leakage_audit.json")
    before = read_json(root_138w / "generated_before_scoring_report.json")
    canary = read_json(root_138w / "expected_output_canary_report.json")
    scan = read_json(root_138w / "ast_shortcut_scan_report.json")
    replay = read_json(root_138w / "determinism_replay_report.json")
    source = read_json(root_138w / "source_checkpoint_integrity_manifest.json")
    target = read_json(root_138w / "target_checkpoint_integrity_manifest.json")
    traces = read_jsonl(root_138w / "raw_generation_trace.jsonl")

    if decision.get("verdict") != "ANSWER_VALUE_GROUNDING_REPAIR_FAILS" or decision.get("decision") != "wrapper_success_without_value_grounding_persists" or decision.get("next") != "138WV_WRAPPER_VALUE_DECOUPLING_FAILURE_ANALYSIS":
        raise GateError("UPSTREAM_138W_ARTIFACT_MISSING", "138W did not route to 138WV")
    required_profile = {
        "answer_prefix_accuracy": 1.0,
        "eval_namespace_emission_accuracy": 1.0,
        "answer_value_accuracy": 0.0,
        "exact_answer_accuracy": 0.0,
        "value_after_prefix_accuracy": 0.0,
        "post_wrapper_garbage_token_rate": 0.0,
        "stale_chat_fragment_rate": 0.0,
        "train_namespace_leak_rate": 0.0,
    }
    mismatches = {key: {"expected": expected, "actual": aggregate.get(key)} for key, expected in required_profile.items() if aggregate.get(key) != expected}
    if mismatches or parrot.get("parrot_trap_detected") is not False:
        raise GateError("UPSTREAM_138W_ARTIFACT_MISSING", "138W metric profile does not match 138WV route", mismatches)
    if canary.get("expected_output_canary_passed") is not True or scan.get("ast_shortcut_scan_passed") is not True:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138W helper canary/AST integrity failed")
    if controls.get("controls_failed") is not True or leakage.get("leakage_rejected") is not True or replay.get("determinism_replay_passed") is not True:
        raise GateError("UPSTREAM_138W_ARTIFACT_MISSING", "138W controls/leakage/determinism missing")
    if source.get("source_checkpoint_unchanged") is not True or target.get("target_checkpoint_changed") is not True:
        raise GateError("UPSTREAM_138W_ARTIFACT_MISSING", "138W checkpoint integrity missing")
    if before.get("generated_text_produced_before_scoring") is not True:
        raise GateError("UPSTREAM_138W_ARTIFACT_MISSING", "138W generated-before-scoring missing")
    for trace in traces:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS or set(request) & FORBIDDEN_HELPER_KEYS:
            raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138W helper request metadata violation")
    stop_reasons = sorted({trace.get("response", {}).get("stop_reason") for trace in traces})
    manifest = {
        "schema_version": "phase_138wv_upstream_138w_manifest_v1",
        "upstream_138w_root": rel(root_138w),
        "verified": True,
        "verdict": decision.get("verdict"),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "helper_integrity_passed": True,
        "canary_passed": True,
        "ast_scan_passed": True,
        "controls_failed": True,
        "leakage_rejected": True,
        "determinism_replay_passed": True,
        "source_checkpoint_unchanged": True,
        "target_checkpoint_changed": True,
        "generated_text_before_scoring": True,
        "no_expected_or_scorer_metadata_reached_helper_requests": True,
        "stop_reasons": stop_reasons,
        "stop_reason_is_max_new_tokens": stop_reasons == ["max_new_tokens"],
        "literal_eos_artifact_present": False,
        **required_profile,
    }
    write_json(out / "upstream_138w_manifest.json", manifest)
    return {"decision": decision, "aggregate": aggregate, "value": value, "parrot": parrot, "carrier": carrier, "manifest": manifest}


def first_answer_marker(text: str) -> re.Match[str] | None:
    return re.search(r"\bANSWER=E", text)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_+-]+|[^\sA-Za-z0-9_+-]", text)


def first_nonspace_token(text: str) -> str | None:
    tokens = tokenize(text.strip())
    return tokens[0] if tokens else None


def value_source_type(family: str, scoring_mode: str | None) -> str:
    if family == "VALUE_DIRECT_COPY_DIAGNOSTIC":
        return "direct_copy"
    if family == "VALUE_RULE_DERIVED":
        return "rule_derived"
    if family == "VALUE_TABLE_DERIVED":
        return "table_derived"
    if family == "VALUE_COMPOSITION_DERIVED":
        return "composition_derived"
    if family == "VALUE_OOD_SYMBOL_BINDING":
        return "ood_symbol"
    return scoring_mode or family.lower()


def extract_candidate(post_wrapper_text: str) -> tuple[str | None, str]:
    token = first_nonspace_token(post_wrapper_text)
    if token is None:
        return None, "no_candidate"
    normalized = token.strip().strip(":;,.").lower()
    if normalized in NEUTRAL_DEFAULTS:
        return token, "neutral_default"
    if normalized in GENERIC_VALUES:
        return token, "generic_value"
    if normalized in STRUCTURAL_TOKENS or token.startswith("ANSWER"):
        return token, "structural_token"
    if token.startswith("TR"):
        return token, "train_seen_value"
    if token.startswith("EV"):
        return token, "eval_value_candidate"
    if re.fullmatch(r"[A-Za-z0-9_+-]+", token):
        return token, "wrong_specific_value"
    return token, "unknown_candidate"


def classify_post_wrapper(post_wrapper_text: str, expected_value: str, generated_text: str, candidate: str | None, candidate_label: str) -> str:
    stripped = post_wrapper_text.strip()
    if not stripped:
        return "immediate_termination_proxy"
    if stripped != post_wrapper_text and not stripped:
        return "empty_or_whitespace_after_wrapper"
    if candidate_label == "neutral_default":
        return "default_neutral_attractor"
    if candidate_label == "structural_token" or re.match(r"(?i)\s*(question|prompt|fact|row|user|assistant|answer|next|case|sample)\b", stripped):
        return "structural_format_echo"
    if candidate_label == "generic_value":
        return "generic_wrong_value"
    if re.search(r"(.)\1{7,}", stripped):
        return "repeated_symbol_or_punctuation"
    if "\ufffd" in stripped or re.search(r"[^A-Za-z0-9_=\s:;.,+\-/]{4,}", stripped):
        return "garbled_after_wrapper"
    expected_before = generated_text.find(expected_value)
    wrapper_pos = generated_text.find("ANSWER=E")
    if expected_before >= 0 and wrapper_pos >= 0 and expected_before < wrapper_pos:
        return "delayed_correct_value_wrong_position"
    if candidate_label in {"train_seen_value", "eval_value_candidate", "wrong_specific_value"}:
        return "wrong_specific_value"
    return "unknown_post_wrapper_behavior"


def build_anatomy_rows(eval_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], scoring_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_by_id = {row["row_id"]: row for row in eval_rows}
    score_by_id = {row["row_id"]: row for row in scoring_rows}
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = eval_by_id[raw["row_id"]]
        score = score_by_id[raw["row_id"]]
        generated = raw["generated_text"]
        marker = first_answer_marker(generated)
        if marker:
            post = generated[marker.end() :]
            expected_pos = post.find(row["answer_value"])
        else:
            post = ""
            expected_pos = -1
        candidate, candidate_label = extract_candidate(post)
        rows.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "seed": row["seed"],
                "generated_text": generated,
                "expected_output": row["expected_output"],
                "expected_value": row["answer_value"],
                "post_wrapper_text": post,
                "post_wrapper_first_token": tokenize(post)[0] if tokenize(post) else None,
                "post_wrapper_first_nonspace_token": first_nonspace_token(post),
                "post_wrapper_length_chars": len(post),
                "post_wrapper_length_tokens_or_bytes": len(post.encode("utf-8", errors="replace")),
                "correct_value_present_after_wrapper": expected_pos >= 0,
                "expected_value_position": expected_pos,
                "generated_value_candidate": candidate,
                "value_candidate_source_label": candidate_label,
                "value_source_type": value_source_type(row["family"], row.get("scoring_mode")),
                "score_failure_reason": score.get("failure_reason"),
                "answer_token": score.get("answer_token"),
            }
        )
    return rows


def taxonomy_rows(anatomy: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in anatomy:
        label = classify_post_wrapper(
            row["post_wrapper_text"],
            row["expected_value"],
            row["generated_text"],
            row["generated_value_candidate"],
            row["value_candidate_source_label"],
        )
        rows.append({**row, "primary_post_wrapper_class": label})
    return rows


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def distribution(taxonomy: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(taxonomy)
    counts = Counter(row["primary_post_wrapper_class"] for row in taxonomy)
    rates = {f"{label}_rate": rate(counts.get(label, 0), total) for label in PRIMARY_LABELS}

    def grouped(group_key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in taxonomy:
            groups[str(row[group_key])].append(row)
        result: dict[str, Any] = {}
        for key, rows in sorted(groups.items()):
            local = Counter(row["primary_post_wrapper_class"] for row in rows)
            result[key] = {f"{label}_rate": rate(local.get(label, 0), len(rows)) for label in PRIMARY_LABELS}
            result[key]["row_count"] = len(rows)
        return result

    return {
        "schema_version": "phase_138wv_attractor_distribution_report_v1",
        "row_count": total,
        **rates,
        "label_counts": {label: counts.get(label, 0) for label in PRIMARY_LABELS},
        "by_family": grouped("family"),
        "by_seed": grouped("seed"),
        "by_value_source_type": grouped("value_source_type"),
    }


def value_candidate_report(anatomy: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(anatomy)
    labels = [
        "no_candidate",
        "neutral_default",
        "generic_value",
        "prompt_copied_value",
        "train_seen_value",
        "eval_expected_value",
        "wrong_specific_value",
        "structural_token",
        "unknown_candidate",
    ]
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in anatomy:
        label = row["value_candidate_source_label"]
        if row["generated_value_candidate"] == row["expected_value"]:
            label = "eval_expected_value"
        elif row["generated_value_candidate"] and row["generated_value_candidate"] in row["generated_text"] and row["generated_value_candidate"] in row.get("post_wrapper_text", ""):
            # Keep the deterministic label assigned by extract_candidate unless it is a direct expected match.
            label = label
        if label not in labels:
            label = "unknown_candidate"
        counts[label] += 1
        rows.append({"row_id": row["row_id"], "family": row["family"], "candidate": row["generated_value_candidate"], "candidate_label": label})
    payload = {"schema_version": "phase_138wv_value_candidate_report_v1", "row_count": total, "rows": rows}
    payload.update({f"{label}_rate": rate(counts.get(label, 0), total) for label in labels})
    payload["expected_value_candidate_rate"] = payload["eval_expected_value_rate"]
    payload["candidate_counts"] = {label: counts.get(label, 0) for label in labels}
    return payload


def parrot_recheck(parrot: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    all_derived_zero = (
        parrot.get("rule_derived_value_accuracy") == 0.0
        and parrot.get("table_derived_value_accuracy") == 0.0
        and parrot.get("composition_derived_value_accuracy") == 0.0
        and parrot.get("ood_symbol_value_accuracy") == 0.0
    )
    return {
        "schema_version": "phase_138wv_parrot_and_derivation_recheck_v1",
        "prompt_value_copy_accuracy": parrot.get("prompt_value_copy_accuracy"),
        "rule_derived_value_accuracy": parrot.get("rule_derived_value_accuracy"),
        "table_derived_value_accuracy": parrot.get("table_derived_value_accuracy"),
        "composition_derived_value_accuracy": parrot.get("composition_derived_value_accuracy"),
        "ood_symbol_value_accuracy": parrot.get("ood_symbol_value_accuracy"),
        "parrot_trap_detected": parrot.get("parrot_trap_detected"),
        "all_derived_metrics_remain_zero": all_derived_zero,
        "value_grounding_absent_confirmed": all_derived_zero and aggregate.get("answer_value_accuracy") == 0.0,
    }


def select_root(distribution_report: dict[str, Any]) -> dict[str, Any]:
    termination_sum = distribution_report["immediate_termination_proxy_rate"] + distribution_report["empty_or_whitespace_after_wrapper_rate"]
    if distribution_report["unknown_post_wrapper_behavior_rate"] > 0.10:
        root = "wrapper_value_decoupling_ambiguous"
        evidence = "unknown_post_wrapper_behavior_rate > 0.10"
    elif termination_sum >= 0.50:
        root = "wrapper_termination_proxy_dominant"
        evidence = "immediate_termination_proxy_rate + empty_or_whitespace_after_wrapper_rate >= 0.50"
    elif distribution_report["default_neutral_attractor_rate"] >= 0.40:
        root = "default_neutral_attractor_dominant"
        evidence = "default_neutral_attractor_rate >= 0.40"
    elif distribution_report["structural_format_echo_rate"] >= 0.30:
        root = "structural_format_echo_dominant"
        evidence = "structural_format_echo_rate >= 0.30"
    elif distribution_report["wrong_specific_value_rate"] >= 0.40:
        root = "wrong_specific_value_attractor_dominant"
        evidence = "wrong_specific_value_rate >= 0.40"
    else:
        root = "mixed_post_wrapper_attractors"
        evidence = "no dominant single attractor above threshold"
    return {
        "schema_version": "phase_138wv_wrapper_value_decoupling_root_cause_v1",
        "root_cause": root,
        "evidence": evidence,
        "evidence_type": "computed_from_artifact",
        "literal_eos_claimed": False,
        "immediate_eos_rate": "not_computed_literal_eos_not_available",
        "immediate_termination_proxy_rate": distribution_report["immediate_termination_proxy_rate"],
        "topological_inhibition_claim_status": "hypothesis_only_diagnostic_gap_without_instrumentation",
    }


def recommend(root: str) -> dict[str, Any]:
    route = {
        "wrapper_termination_proxy_dominant": "138X_WRAPPER_TERMINATION_VALUE_PATH_REPAIR_PLAN",
        "default_neutral_attractor_dominant": "138Y_VALUE_ACTIVATION_CARRIER_REDESIGN_PLAN",
        "structural_format_echo_dominant": "138Z_SEMANTIC_VS_FORMAT_OBJECTIVE_REBALANCE_PLAN",
        "wrong_specific_value_attractor_dominant": "138U_WRONG_VALUE_ATTRACTOR_ANALYSIS",
        "mixed_post_wrapper_attractors": "138WB_WRAPPER_VALUE_DECOUPLING_MANUAL_REVIEW_PACKET",
        "wrapper_value_decoupling_ambiguous": "138WB_WRAPPER_VALUE_DECOUPLING_MANUAL_REVIEW_PACKET",
    }
    return {
        "schema_version": "phase_138wv_next_repair_recommendation_v1",
        "root_cause": root,
        "recommended_next": route[root],
        "clean_negative_accepted": True,
        "no_model_fix_performed": True,
    }


def make_decision(root: dict[str, Any], recommendation: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if root["root_cause"] == "wrapper_value_decoupling_ambiguous":
        decision_name = "wrapper_value_decoupling_ambiguous"
        next_step = "138WB_WRAPPER_VALUE_DECOUPLING_MANUAL_REVIEW_PACKET"
    else:
        decision_name = "wrapper_value_decoupling_failure_analysis_complete"
        next_step = recommendation["recommended_next"]
    decision = {
        "schema_version": "phase_138wv_decision_v1",
        "decision": decision_name,
        "next": next_step,
        "verdict": "WRAPPER_VALUE_DECOUPLING_FAILURE_ANALYSIS_COMPLETE",
        "root_cause": root["root_cause"],
        "artifact_only": True,
        "new_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "literal_eos_claimed": False,
        "immediate_termination_is_proxy_not_literal_eos": True,
        "topological_inhibition_claim_status": "hypothesis_only_diagnostic_gap_without_instrumentation",
        **FALSE_FLAGS,
    }
    verdicts = [
        decision["verdict"],
        "ARTIFACT_ONLY_ANALYSIS",
        "LITERAL_EOS_NOT_CLAIMED",
        "POST_WRAPPER_ATTRACTOR_CLASSIFIED",
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
    else:
        decision_name = "upstream_138w_artifact_missing"
        next_step = "138WV_UPSTREAM_138W_ARTIFACT_MISSING"
    decision = {"schema_version": "phase_138wv_failure_decision_v1", "decision": decision_name, "next": next_step, "verdict": error.verdict, "failure_message": error.message, "literal_eos_claimed": False, **FALSE_FLAGS}
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [error.verdict], decision, error.message)
    write_report(out, [error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {"schema_version": "phase_138wv_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["WRAPPER_VALUE_DECOUPLING_ANALYSIS_RUNNING"], {"decision": "pending", "next": "pending"})

    root_138w = resolve_path(args.upstream_138w_root)
    upstream = verify_upstream(out, root_138w)
    append_progress(out, "upstream verification", verified=True)
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_138wv_analysis_config_v1",
            "artifact_only": True,
            "new_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            "literal_eos_claimed": False,
            "immediate_termination_proxy_not_literal_eos": True,
            "stop_reason_expected": "max_new_tokens",
            "topological_inhibition_claim_status": "hypothesis_only_diagnostic_gap_without_instrumentation",
        },
    )
    append_progress(out, "artifact loading", upstream_138w_root=rel(root_138w))

    eval_rows = read_jsonl(root_138w / "eval_rows.jsonl")
    raw_rows = read_jsonl(root_138w / "raw_generation_results.jsonl")
    scoring_rows = read_jsonl(root_138w / "scoring_results.jsonl")
    anatomy = build_anatomy_rows(eval_rows, raw_rows, scoring_rows)
    write_json(out / "post_wrapper_value_anatomy_report.json", {"schema_version": "phase_138wv_post_wrapper_value_anatomy_report_v1", "row_count": len(anatomy), "rows": anatomy})
    append_progress(out, "post-wrapper anatomy analysis", row_count=len(anatomy))
    refresh_status(out, "running", ["POST_WRAPPER_ANATOMY_ANALYZED"], {"decision": "pending", "next": "pending"})

    taxonomy = taxonomy_rows(anatomy)
    counts = Counter(row["primary_post_wrapper_class"] for row in taxonomy)
    write_json(out / "silence_taxonomy_report.json", {"schema_version": "phase_138wv_silence_taxonomy_report_v1", "row_count": len(taxonomy), "primary_label_counts": dict(counts), "exactly_one_primary_label_per_row": all(row["primary_post_wrapper_class"] in PRIMARY_LABELS for row in taxonomy), "literal_eos_claimed": False, "rows": taxonomy})
    append_progress(out, "silence taxonomy", labels=dict(counts))

    dist = distribution(taxonomy)
    write_json(out / "attractor_distribution_report.json", dist)
    append_progress(out, "attractor distribution", wrong_specific_value_rate=dist["wrong_specific_value_rate"], immediate_termination_proxy_rate=dist["immediate_termination_proxy_rate"])

    candidates = value_candidate_report(anatomy)
    write_json(out / "value_candidate_report.json", candidates)
    append_progress(out, "value candidate analysis", wrong_specific_value_rate=candidates["wrong_specific_value_rate"], train_seen_value_rate=candidates["train_seen_value_rate"])

    parrot = parrot_recheck(upstream["parrot"], upstream["aggregate"])
    write_json(out / "parrot_and_derivation_recheck.json", parrot)
    append_progress(out, "parrot/derivation recheck", value_grounding_absent_confirmed=parrot["value_grounding_absent_confirmed"])

    root = select_root(dist)
    write_json(out / "wrapper_value_decoupling_root_cause.json", root)
    append_progress(out, "root cause selection", root_cause=root["root_cause"])

    recommendation = recommend(root["root_cause"])
    write_json(out / "next_repair_recommendation.json", recommendation)
    append_progress(out, "recommendation", next=recommendation["recommended_next"])

    write_json(
        out / "diagnostic_gap_register.json",
        {
            "schema_version": "phase_138wv_diagnostic_gap_register_v1",
            "gaps": [
                {"field": "literal_eos_evidence", "status": "diagnostic_gap", "reason": "138W helper stop_reason is max_new_tokens, not EOS"},
                {"field": "hidden_state_topological_inhibition", "status": "diagnostic_gap", "reason": "138WV does not inspect activations or graph topology"},
            ],
        },
    )
    write_json(
        out / "risk_register.json",
        {
            "schema_version": "phase_138wv_risk_register_v1",
            "risks": [
                {"risk": "immediate termination proxy is misread as literal EOS", "mitigation": "literal_eos_claimed=false and stop_reason=max_new_tokens recorded"},
                {"risk": "wrong-specific value attractor is hidden by aggregate value accuracy", "mitigation": "post-wrapper candidate taxonomy records generated candidates for every row"},
            ],
        },
    )
    decision, verdicts = make_decision(root, recommendation)
    write_json(out / "decision.json", decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    refresh_status(out, "completed", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_138wv_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138w-root", default=str(DEFAULT_UPSTREAM_138W_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except GateError as exc:
        write_failure_decision(args, exc)
        print(f"138WV failed closed: {exc.verdict}: {exc.message}")
        return 1 if exc.verdict == "138WV_BOUNDARY_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
